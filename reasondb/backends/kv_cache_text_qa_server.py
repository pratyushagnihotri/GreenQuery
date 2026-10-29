import argparse
import json
import asyncio
import logging
import time
from tqdm import tqdm
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List


from reasondb.backends.kv_cache_base import KVCachingBackendBase
from reasondb.backends.text_qa import PORT_KV_TEXT_QA
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress, KVzipPress

from transformers import DynamicCache, pipeline  # type: ignore
import os
import sys as _sys
import contextlib

# _APE_PATH = os.path.join(os.path.dirname(__file__), "../../APE")
# if _APE_PATH not in _sys.path:
#     _sys.path.insert(0, _APE_PATH)
# from ape import enable_attention_prefill_prefix, enable_attention_prefill_query  # noqa: E402
# from ape.ape_llama_varlen import enable_llama_attention_prefill_query_varlen  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel  # noqa: E402

from reasondb.memory_footprint.memory_report import compute_memory_footprints


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
    "kvzip": lambda compression_ratio: KVzipPress(compression_ratio=compression_ratio),
}


class KvTextQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id: int,
        compression_ratios=(0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9),
        # compression_ratios=(0.5, 0.6, 0.8, 0.9),
        batch_sizes=None,
        press_name="expected_attention",
        gold_vanilla=False,
    ):
        self.device_id = device_id
        self.compression_ratios = compression_ratios
        batch_sizes = batch_sizes or (None,) * len(compression_ratios)
        self.compression_ratio_to_batch_size = {
            cr: bs for cr, bs in zip(compression_ratios, batch_sizes)
        }
        self.model_name = model_name
        self.press_name = press_name
        self.gold_vanilla = gold_vanilla
        self.init()

    def compute_text_qa_response(
        self,
        column_name: str,
        texts: List[str],
        questions: List[str],
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ):
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        return asyncio.run(
            self._run_kv_cache_text(
                column_name=column_name,
                texts=texts,
                all_questions=questions,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
            )
        )

    def compute_text_qa_join_response(
        self,
        column_name: str,
        unique_texts: List[str],
        questions_per_text: List[List[str]],
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ):
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        return asyncio.run(
            self._run_kv_cache_text_join(
                column_name=column_name,
                unique_texts=unique_texts,
                questions_per_text=questions_per_text,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
            )
        )

    def compute_direct_qa_response(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        batch_size: int = 128,
    ) -> dict:
        """Run direct LLM inference. contexts[i] is the left-side text prepended before the instruction."""
        return asyncio.run(
            self._run_direct_qa(
                questions=questions,
                contexts=contexts,
                boolean_question=boolean_question,
                batch_size=batch_size,
            )
        )

    def compute_generation_response(self, messages: List[dict], max_new_tokens: int = 512) -> dict:
        """Generate one text response from the already-loaded local model."""
        assert self.pipe is not None and self.tokenizer is not None
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("Local generation accepts text-only messages")
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        device = next(self.pipe.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started_at = time.time()
        with torch.no_grad():
            generated = self.pipe.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finished_at = time.time()
        output_ids = generated[0, input_tokens:]
        return {
            "text": self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            "telemetry": {
                "model_name": self.model_name,
                "model_revision": getattr(self.pipe.model.config, "_commit_hash", None),
                "request_count": 1,
                "processed_rows": 1,
                "answer_cache_hits": 0,
                "kv_cache_hits": 0,
                "input_tokens": input_tokens,
                "output_tokens": int(output_ids.shape[0]),
                "inference_started_at": started_at,
                "inference_finished_at": finished_at,
                "inference_seconds": finished_at - started_at,
                "gpu_index": self.device_id,
                "execution_mode": "local_planner_generation",
            },
        }

    async def _run_direct_qa(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        batch_size: int = 128,
    ) -> dict:
        """Evaluate questions without KV caches. Each prompt is: context + instruction + question.

        This mirrors the token structure used by TextQaFilter (KV cache join), where the left
        document occupies the KV cache and the instruction+question follow it.  Here the left
        document (context) is placed inline before the instruction inside the same user turn.
        Questions are processed in batches of batch_size for GPU efficiency.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."
        assert (
            len(questions) == len(contexts)
        ), f"questions ({len(questions)}) and contexts ({len(contexts)}) must have the same length"

        max_new_tokens = 4 if boolean_question else 64
        if boolean_question:
            instruction = (
                "Answer '1' if the context above satisfies the following condition, '0' otherwise. "
                "Do not add any other comments."
            )
        else:
            instruction = "Answer the following question based on both the context above and the following text. Do not add any other comments."

        answer_prefix = "Answer: "

        first_device = next(self.pipe.model.parameters()).device
        id0 = self.tokenizer.convert_tokens_to_ids("0")
        id1 = self.tokenizer.convert_tokens_to_ids("1")

        # Build all full prompt strings first
        full_prompts = []
        for context, question in zip(contexts, questions):
            full_content = context + instruction + " " + question
            if self.tokenizer.chat_template is None:
                full_prompts.append(full_content + "\n" + answer_prefix)
            else:
                full_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": full_content}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    + answer_prefix
                )

        inference_started_at = time.time()
        answers = []
        log_odds_list = []
        row_telemetry = []
        input_tokens_total = 0
        output_tokens_total = 0

        inference_batch_count = 0
        for batch_start in tqdm(
            range(0, len(full_prompts), batch_size),
            desc="Processing direct QA (batched)",
        ):
            inference_batch_count += 1
            batch_prompts = full_prompts[batch_start : batch_start + batch_size]
            # tokenizer handles left-padding automatically (padding_side="left" set in init)
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()
            input_len = inputs["input_ids"].shape[1]
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            logits = generated.scores[0]  # (B, vocab)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_odds_batch = (log_probs[:, id1] - log_probs[:, id0]).cpu().tolist()
            decoded_batch = self.tokenizer.batch_decode(
                generated.sequences[:, input_len:],
                skip_special_tokens=True,
            )
            generated_tokens = generated.sequences[:, input_len:]
            output_token_counts = (
                (generated_tokens != self.tokenizer.pad_token_id).sum(dim=1).tolist()
            )
            answers.extend(decoded_batch)
            log_odds_list.extend(log_odds_batch)
            input_tokens_total += sum(int(value) for value in input_token_counts)
            output_tokens_total += sum(int(value) for value in output_token_counts)
            for offset, (prompt, answer, log_odd, input_count, output_count) in enumerate(
                zip(
                    batch_prompts,
                    decoded_batch,
                    log_odds_batch,
                    input_token_counts,
                    output_token_counts,
                )
            ):
                row_telemetry.append(
                    {
                        "row_offset": batch_start + offset,
                        "prompt": prompt,
                        "answer": answer,
                        "log_odds": float(log_odd),
                        "input_tokens": int(input_count),
                        "output_tokens": int(output_count),
                    }
                )
            del inputs, generated
            torch.cuda.empty_cache()

        inference_finished_at = time.time()
        model_revision = getattr(self.pipe.model.config, "_commit_hash", None)
        gpu_name = (
            torch.cuda.get_device_name(self.device_id)
            if torch.cuda.is_available()
            else None
        )
        return {
            "answers": answers,
            "log_odds": log_odds_list,
            "telemetry": {
                "model_name": self.model_name,
                "model_revision": model_revision,
                "execution_mode": "direct_no_cache",
                "inference_batch_size": batch_size,
                "inference_batch_count": inference_batch_count,
                "request_count": 1,
                "processed_rows": len(contexts),
                "answer_cache_hits": 0,
                "kv_cache_hits": 0,
                "input_tokens": input_tokens_total,
                "output_tokens": output_tokens_total,
                "inference_started_at": inference_started_at,
                "inference_finished_at": inference_finished_at,
                "inference_seconds": inference_finished_at - inference_started_at,
                "gpu_index": self.device_id,
                "gpu_model": gpu_name,
                "rows": row_telemetry,
            },
        }

    def hash_text(self, text: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()

    def _unique_layer_devices(self) -> List[torch.device]:
        """Return deduplicated list of GPU devices that hold model layers (ordered)."""
        seen: set = set()
        devices: List[torch.device] = []
        for layer in self.pipe.model.model.layers:
            d = layer.self_attn.q_proj.weight.device
            key = (d.type, d.index)
            if key not in seen:
                seen.add(key)
                devices.append(d)
        return devices

    async def prepare_caches(
        self,
        column_name: str,
        texts: List[str],
        cache_dir: str,
        compression_ratio: float,
    ):
        # Gold-vanilla mode runs the model directly and therefore needs no KV cache.
        if self.gold_vanilla and compression_ratio == 0.0:
            logger.info(
                "Skipping cache preparation for gold-vanilla mode (CR=0)"
            )
            return

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"

        # Check if caches exist and generate if not (current method: checked one by one)
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        errors = {}
        cache_filenames = []

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches {compression_ratio}",
        ):
            text = "" if text is None else str(text)
            hash_name = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                await self._generate_cache_for_text(
                    text, cache_filename, compression_ratio
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)

        compute_memory_footprints(
            cache_dir, column_name, cache_filenames, model_name=self.model_name
        )
        with open(
            f"{save_dir}/ERRORS.json",
            "w",
        ) as f:
            json.dump(errors, f, indent=4)

    def to_compression_tag(self, compression_ratio: float) -> str:
        """Convert compression ratio to a string tag for directory naming."""
        return (
            str(compression_ratio).replace(".", "_")
            if compression_ratio != 0.0
            else "0"
        )

    def init(self):
        """Initialize the text model pipeline and compression settings."""
        logger.info("Setting up KV Cache Text Filter...")

        # Set up device
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"

        # Initialize the pipeline
        args = [{"attn_implementation": "sdpa"}, {}]
        self.pipe = None
        for x in args:
            try:
                self.pipe = pipeline(
                    "kv-press-text-generation",  # type: ignore
                    model=self.model_name,
                    # device=self.device,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=x,  # type: ignore
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing model with args {x}: {str(type(e))}({str(e)})",
                    exc_info=True,
                )

        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Set up compression press
        if self.press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{self.press_name}'. Available options: {list(PRESS.keys())}"
            )
        self.presses = {
            cr: KeyRerotationPress(PRESS[self.press_name](compression_ratio=cr))
            if cr > 0.0
            else None
            for cr in self.compression_ratios
        }

        logger.info(
            f"Using press {self.press_name} with compression_ratios {self.compression_ratios}",
        )
        logger.info(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

    async def _run_vanilla_inference(
        self,
        texts: List[str],
        all_questions: List[str],
        boolean_question: bool,
    ):
        """Run vanilla inference without KV caching."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64

        answers = []
        log_odds = []
        id0 = self.tokenizer.convert_tokens_to_ids("0")
        id1 = self.tokenizer.convert_tokens_to_ids("1")

        # Process texts one by one (could be batched for efficiency)
        for text, question in tqdm(
            zip(texts, all_questions),
            total=len(texts),
            desc="Processing texts (vanilla)",
        ):
            try:
                text = "" if text is None else str(text)
                question = "" if question is None else str(question)

                # Determine question suffix based on chat template
                if self.tokenizer.chat_template is None:
                    question_suffix = "\n"
                else:
                    template_context = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Example of context\n###"}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    _, question_suffix = template_context.split("\n###")

                # Create full prompt
                full_prompt = text + "\n" + question + question_suffix + answer_prefix

                # Tokenize input
                inputs = self.tokenizer(full_prompt, return_tensors="pt")

                # Move inputs to device
                first_device = next(self.pipe.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                # Decode the generated tokens
                decoded = self.tokenizer.decode(
                    generated.sequences[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                first_token_log_probs = torch.nn.functional.log_softmax(
                    generated.scores[0][0], dim=-1
                )

                answers.append(decoded)
                log_odds.append(
                    (first_token_log_probs[id1] - first_token_log_probs[id0])
                    .cpu()
                    .item()
                )

                # Cleanup
                del inputs, generated
                torch.cuda.empty_cache()

            except Exception as e:
                logger.exception(
                    "Vanilla inference failed for question %r: %s", question, e
                )
                raise

        return {"answers": answers, "log_odds": log_odds}

    async def _run_kv_cache_text(
        self,
        column_name: str,
        texts: List[str],
        all_questions: List[str],
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
    ):
        """Run KV cache-based text inference on the data."""

        # Use direct model inference for the uncompressed gold-vanilla backend.
        if self.gold_vanilla and compression_ratio == 0.0:
            logger.info("Using gold-vanilla mode for inference (CR=0)")
            return await self._run_vanilla_inference(
                texts=texts,
                all_questions=all_questions,
                boolean_question=boolean_question,
            )

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        # Check if caches exist and generate if not
        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        cache_files = []
        hashed_texts = []

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches for cr {compression_ratio}",
        ):
            hashed_text = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hashed_text}.pt"

            assert os.path.exists(
                cache_filename
            ), f"Cache file does not exist for text at index {i}: {cache_filename}"
            cache_files.append(cache_filename)
            hashed_texts.append(hashed_text)

        if not cache_files:
            logger.warning("No valid caches or texts found")
            return []

        # Set up text processing parameters
        if boolean_question:
            context = "Answer the following question based on the context with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the context. Do not add any other comments."
        all_questions = [context + " " + q for q in all_questions]
        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64
        # Release any fragmented reserved-but-unallocated GPU memory before
        # measuring free memory for batch size estimation.
        torch.cuda.empty_cache()
        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        # Peek at the first cache file to get context token count, then tokenize
        # the longest question to estimate query-token activation memory.
        first_cache_data = torch.load(
            cache_files[0], map_location="cpu", weights_only=False
        )
        if isinstance(first_cache_data, dict):
            _ctx_tokens = first_cache_data["key_cache"][0].shape[2]
        else:
            _ctx_tokens = first_cache_data.key_cache[0].shape[2]
        del first_cache_data
        longest_q = max(all_questions, key=len)
        _q_tokens = len(
            self.tokenizer.encode(longest_q, add_special_tokens=False)
        ) + max_new_tokens
        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            layer_devices=self._unique_layer_devices(),
            cache_dir=cache_dir,
            file_paths=cache_files,
            max_question_tokens=_q_tokens,
            max_context_tokens=_ctx_tokens,
        )

        answers = []
        log_odds = []

        # Process texts in batches
        for batch_start in tqdm(
            range(0, len(cache_files), batch_size),
            desc=f"Processing batches for CR {compression_ratio}",
        ):
            batch_input_ids = []
            batch_attention_masks = []
            caches = []
            context_lengths = []
            questions = []

            batch_cache_files = cache_files[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_questions = all_questions[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_size_actual = len(batch_cache_files)

            # Generate or load caches for each text in the batch
            # Pre-compute layer devices once (handles multi-GPU tensor parallel models)
            layer_devices = [
                self.pipe.model.model.layers[i].self_attn.q_proj.weight.device
                for i in range(len(self.pipe.model.model.layers))
            ]
            _t_disk_start = time.time()
            for i in range(batch_size_actual):
                # Load to CPU first, then immediately move each layer to its target GPU device
                # so that subsequent pad/cat ops happen on GPU (much faster than CPU)
                cache_data = torch.load(
                    batch_cache_files[i], map_location="cpu", weights_only=False
                )
                # Support both old DynamicCache objects and new plain-dict format
                if isinstance(cache_data, dict):
                    key_cache = cache_data["key_cache"]
                    value_cache = cache_data["value_cache"]
                else:
                    key_cache = cache_data.key_cache
                    value_cache = cache_data.value_cache
                key_cache = [k.to(layer_devices[j]) for j, k in enumerate(key_cache)]
                value_cache = [
                    v.to(layer_devices[j]) for j, v in enumerate(value_cache)
                ]
                caches.append((key_cache, value_cache))
                context_lengths.append(key_cache[0].shape[2])
                questions.append(batch_questions[i])
            _t_disk_end = time.time()

            if not caches:
                continue

            max_context_len = max(context_lengths)

            question_ids_list = []
            context_ids_list = []
            padded_context_ids_mask_list = []

            _t_cpu_batch_start = time.time()
            for i, (ctx_len, q) in enumerate(zip(context_lengths, questions)):
                # Pad context to align with KV cache
                padded_context_ids = torch.full(
                    (1, ctx_len),
                    self.tokenizer.pad_token_id + 1,
                    device=self.device,
                )
                pad_len = max_context_len - ctx_len
                padding_ids = torch.full(
                    (1, pad_len), self.tokenizer.pad_token_id, device=self.device
                )
                padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)
                padding_mask = torch.zeros_like(padding_ids)
                padded_context_mask = torch.ones_like(padded_context_ids)
                padded_context_ids_mask = torch.cat(
                    [padding_mask, padded_context_mask], dim=1
                )

                # Determine question suffix
                if self.tokenizer.chat_template is None:
                    question_suffix = "\n"
                else:
                    separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                    template_context = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Example of context" + separator}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    _, question_suffix = template_context.split(separator)

                # Tokenize question (variable-length)
                complete_question = q + question_suffix + answer_prefix
                question_ids = self.tokenizer.encode(
                    complete_question,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                # Store for later padding
                context_ids_list.append(padded_context)
                question_ids_list.append(question_ids)
                padded_context_ids_mask_list.append(padded_context_ids_mask)

            # Compute max question length across batch
            max_question_len = max(q.shape[1] for q in question_ids_list)

            # Pad questions and build batch tensors
            batch_input_ids = []
            batch_attention_masks = []

            for padded_context, question_ids, padded_context_ids_mask in zip(
                context_ids_list, question_ids_list, padded_context_ids_mask_list
            ):
                q_len = question_ids.shape[1]
                q_pad_len = max_question_len - q_len

                # Pad question to match longest in batch
                if q_pad_len > 0:
                    q_padding = torch.full(
                        (1, q_pad_len),
                        self.tokenizer.pad_token_id,
                        device=self.device,
                    )
                    padded_question = torch.cat([q_padding, question_ids], dim=1)
                    padded_question_mask = torch.cat(
                        [torch.zeros_like(q_padding), torch.ones_like(question_ids)],
                        dim=1,
                    )
                else:
                    padded_question = question_ids
                    padded_question_mask = torch.ones_like(question_ids)

                # Concatenate full input
                input_ids = torch.cat([padded_context, padded_question], dim=1)

                # Build attention mask: padding + context + padding + question
                attention_mask = torch.cat(
                    [padded_context_ids_mask, padded_question_mask], dim=1
                )

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)

            # Stack batched tensors
            batched_inputs = torch.cat(batch_input_ids, dim=0)
            batched_attention_mask = torch.cat(batch_attention_masks, dim=0)

            # Batch the caches
            batched_cache = []
            for layer_kv_pairs in zip(*[zip(kc, vc) for kc, vc in caches]):
                max_seq_len = max(k.shape[2] for k, _ in layer_kv_pairs)
                keys_padded = []
                values_padded = []

                for k, v in layer_kv_pairs:
                    seq_len = k.shape[2]
                    pad_len = max_seq_len - seq_len
                    k_padded = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else k
                    )
                    v_padded = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else v
                    )
                    k_padded = k_padded.contiguous()
                    v_padded = v_padded.contiguous()
                    keys_padded.append(k_padded)
                    values_padded.append(v_padded)

                keys_cat = torch.cat(keys_padded, dim=0)
                values_cat = torch.cat(values_padded, dim=0)
                batched_cache.append((keys_cat, values_cat))

            padded_cache = DynamicCache()
            for layer_idx, (keys, values) in enumerate(batched_cache):
                padded_cache.update(keys, values, layer_idx)
            _t_cpu_batch_end = time.time()

            # Move inputs to the device of the embedding layer (first layer)
            # KV caches are already on their target devices (moved during loading)
            _t_gpu_transfer_start = time.time()
            first_device = next(self.pipe.model.parameters()).device
            batched_inputs = batched_inputs.to(first_device)
            batched_attention_mask = batched_attention_mask.to(first_device)
            _t_gpu_transfer_end = time.time()

            # Generate responses
            _t_generate_start = time.time()
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    input_ids=batched_inputs,
                    attention_mask=batched_attention_mask,
                    past_key_values=padded_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            _t_generate_end = time.time()
            logger.info(
                f"[text_qa batch] n={batch_size_actual} "
                f"disk_load={_t_disk_end - _t_disk_start:.2f}s "
                f"cpu_batch={_t_cpu_batch_end - _t_cpu_batch_start:.2f}s "
                f"gpu_transfer={_t_gpu_transfer_end - _t_gpu_transfer_start:.2f}s "
                f"generate={_t_generate_end - _t_generate_start:.2f}s "
                f"total={_t_generate_end - _t_disk_start:.2f}s"
            )

            logits = generated.scores[0]  # type: ignore
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            id0 = self.tokenizer.convert_tokens_to_ids("0")
            id1 = self.tokenizer.convert_tokens_to_ids("1")
            log_probs_0 = log_probs[:, id0]
            log_probs_1 = log_probs[:, id1]
            log_odds_1_vs_0 = log_probs_1 - log_probs_0

            # Decode the generated tokens
            decoded = self.tokenizer.batch_decode(
                generated.sequences[:, batched_inputs.shape[1] :],  # type: ignore
                skip_special_tokens=True,
            )

            answers.extend(decoded)
            log_odds.extend(log_odds_1_vs_0.cpu().tolist())

            # Cleanup batch resources
            del (
                caches,
                batched_inputs,
                batched_attention_mask,
                padded_cache,
                generated,
                logits,
                log_probs,
            )
            del (
                batch_input_ids,
                batch_attention_masks,
                context_ids_list,
                question_ids_list,
                padded_context_ids_mask_list,
                batched_cache,
            )
            torch.cuda.empty_cache()

        # Process answers and create mask
        # mask = []
        # for data_id, answer in zip(data_ids, answers):
        #     predicted_answer = keep_answer.strip().lower() in answer.lower()
        #     # logger.info(f"Row {data_id} answer: {answer} -> {predicted_answer}")
        #     mask.append((data_id, predicted_answer))

        return {"answers": answers, "log_odds": log_odds}

    async def _run_kv_cache_text_join(
        self,
        column_name: str,
        unique_texts: List[str],
        questions_per_text: List[List[str]],
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
    ):
        """Load left KV caches in batches, answer all right-side questions against each batch.

        For N unique left texts and M right-side questions: instead of N×M forward passes of
        batch_size=1, this runs (N/B)×M forward passes of batch_size=B — same total work but
        B× better GPU utilization.

        All questions_per_text[i] are identical (the question template only substitutes right-side
        columns, so question_j is independent of the left text). We use questions_per_text[0] as
        the canonical question list.
        """

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."
        assert len(unique_texts) == len(questions_per_text)
        assert len(questions_per_text) > 0, "questions_per_text is empty"
        canonical_questions = questions_per_text[0]
        questions_are_uniform = all(
            qs == canonical_questions for qs in questions_per_text
        )

        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        if boolean_question:
            context_prefix = "Answer the following question based on the context above with '1' or '0'. Do not add any other comments."
        else:
            context_prefix = "Answer the following question based on both the context above. Do not add any other comments."
        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64
        id0 = self.tokenizer.convert_tokens_to_ids("0")
        id1 = self.tokenizer.convert_tokens_to_ids("1")
        first_device = next(self.pipe.model.parameters()).device

        N = len(unique_texts)
        max_M = max(len(qs) for qs in questions_per_text)
        logger.info(
            f"[join] Starting: N={N} unique left contexts, max_M={max_M} right-side questions, "
            f"compression_ratio={compression_ratio}"
        )
        logger.info(f"[join] Sample questions: {canonical_questions[:3]}")

        # Pre-allocate result arrays (indexed by [left_idx][question_idx]).
        # Each text may have a different number of questions in the ragged (sampled) case.
        answers_per_text = [[""] * len(qs) for qs in questions_per_text]
        log_odds_per_text = [[0.0] * len(qs) for qs in questions_per_text]

        cache_file_paths = [
            f"{save_dir}/cache_entry_{self.hash_text(t)}.pt" for t in unique_texts
        ]
        # Estimate question token count using the longest canonical question.
        # Each entry already contains the right-side raw text substituted in, so
        # the longest string gives the worst-case token count for memory planning.
        longest_question = max(canonical_questions, key=len)
        sample_full_question = context_prefix + " " + longest_question
        sample_q_tokens = len(
            self.tokenizer.encode(sample_full_question, add_special_tokens=False)
        )

        # Load the first cache file to get a representative context token count
        # (cheap CPU load; used only to set the scale factor, not for inference).
        _sample_cache = torch.load(
            cache_file_paths[0], map_location="cpu", weights_only=False
        )
        if isinstance(_sample_cache, dict):
            sample_ctx_tokens = _sample_cache["key_cache"][0].shape[2]
        else:
            sample_ctx_tokens = _sample_cache.key_cache[0].shape[2]
        del _sample_cache

        batch_size = self._get_max_batch_size_join(
            column_name=column_name,
            batch_size=self.compression_ratio_to_batch_size[compression_ratio],
            compression_ratio=compression_ratio,
            layer_devices=self._unique_layer_devices(),
            cache_dir=cache_dir,
            file_paths=cache_file_paths,
            max_question_tokens=sample_q_tokens,
            max_context_tokens=sample_ctx_tokens,
        )

        for batch_start in tqdm(
            range(0, N, batch_size),
            desc="TextQA join: processing batches of left contexts",
        ):
            batch_texts = unique_texts[batch_start : batch_start + batch_size]
            B = len(batch_texts)

            # ----------------------------------------------------------------
            # Step 1: Load B KV caches from disk, move each layer to its target GPU device
            # ----------------------------------------------------------------
            # Pre-compute layer devices once (handles multi-GPU tensor parallel models)
            layer_devices = [
                self.pipe.model.model.layers[i].self_attn.q_proj.weight.device
                for i in range(len(self.pipe.model.model.layers))
            ]
            caches = []  # list of (key_cache, value_cache) on target GPU devices
            context_lens = []
            _t_load_start = time.time()
            for text in batch_texts:
                cache_filename = f"{save_dir}/cache_entry_{self.hash_text(text)}.pt"
                assert os.path.exists(
                    cache_filename
                ), f"Cache file does not exist: {cache_filename}"
                cache_data = torch.load(
                    cache_filename, map_location="cpu", weights_only=False
                )
                if isinstance(cache_data, dict):
                    key_cache = cache_data["key_cache"]
                    value_cache = cache_data["value_cache"]
                else:
                    key_cache = cache_data.key_cache
                    value_cache = cache_data.value_cache
                # Load to CPU first, then immediately move each layer to its target GPU device
                # so that subsequent pad/cat ops happen on GPU (much faster than CPU)
                key_cache = [k.to(layer_devices[j]) for j, k in enumerate(key_cache)]
                value_cache = [
                    v.to(layer_devices[j]) for j, v in enumerate(value_cache)
                ]
                caches.append((key_cache, value_cache))
                ctx_len = key_cache[0].shape[2]
                context_lens.append(ctx_len)
                logger.info(
                    f"[join] Loaded KV cache | text='{text[:80]}{'...' if len(text) > 80 else ''}' | "
                    f"ctx_tokens={ctx_len} | file={os.path.basename(cache_filename)}"
                )
            logger.info(
                f"[join] Batch {batch_start // batch_size + 1}: loaded {B} KV caches in "
                f"{time.time() - _t_load_start:.2f}s | ctx_lens={context_lens}"
            )

            max_ctx = max(context_lens)

            # ----------------------------------------------------------------
            # Step 2: Pad and stack B caches on GPU (already on target devices from Step 1).
            # Shape after stacking: (B, num_heads, max_ctx, head_dim) per layer.
            # The GPU tensors serve as the reference; each question clones from
            # them (VRAM→VRAM copy, no PCIe transfer per question).
            # ----------------------------------------------------------------
            batched_gpu_keys = []
            batched_gpu_values = []
            for layer_idx, layer_kv_pairs in enumerate(
                zip(*[zip(kc, vc) for kc, vc in caches])
            ):
                keys_padded, values_padded = [], []
                for k, v in layer_kv_pairs:
                    pad_len = max_ctx - k.shape[2]
                    k_pad = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else k
                    )
                    v_pad = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else v
                    )
                    keys_padded.append(k_pad)
                    values_padded.append(v_pad)
                # Tensors are already on their target GPU device (moved in Step 1)
                batched_gpu_keys.append(torch.cat(keys_padded, dim=0))
                batched_gpu_values.append(torch.cat(values_padded, dim=0))

            # ----------------------------------------------------------------
            # Step 3: For each question position j, run one forward pass.
            #
            # Fast path (uniform): all B items share the same question — one tokenization,
            # no extra padding, all B KV caches used together.
            #
            # Ragged path: items may have different questions or different numbers of
            # questions.  Only items that have a j-th question form the batch (valid_bs).
            # Questions of different token lengths are right-padded to max_q_len within
            # the batch; the padding tokens get attention_mask = 0.
            # ----------------------------------------------------------------
            _t_questions_start = time.time()
            batch_qs = questions_per_text[batch_start : batch_start + batch_size]
            batch_max_j = max(len(qs) for qs in batch_qs)

            if questions_are_uniform:
                question_iter = enumerate(canonical_questions)
            else:
                question_iter = ((j, None) for j in range(batch_max_j))

            for j, _uniform_question in question_iter:
                if questions_are_uniform:
                    # All items have the same question — tokenize once
                    full_question = context_prefix + " " + _uniform_question
                    logger.info(
                        f"[join] Question {j+1}/{batch_max_j} | "
                        f"prompt='{full_question[:200]}{'...' if len(full_question) > 200 else ''}'"
                    )
                    valid_bs = list(range(B))
                    per_item_questions = [_uniform_question] * B

                    batch_input_ids = []
                    batch_attn_mask = []
                    for ctx_len in context_lens:
                        pad_len = max_ctx - ctx_len
                        if self.tokenizer.chat_template is None:
                            question_suffix = "\n"
                        else:
                            separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                            template_context = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "user",
                                        "content": "Example of context" + separator,
                                    }
                                ],
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                            _, question_suffix = template_context.split(separator)
                        complete_question = (
                            full_question + question_suffix + answer_prefix
                        )
                        question_ids = self.tokenizer.encode(
                            complete_question,
                            return_tensors="pt",
                            add_special_tokens=False,
                        ).to(first_device)
                        ctx_ids = torch.full(
                            (1, ctx_len),
                            self.tokenizer.pad_token_id + 1,
                            device=self.device,
                        )
                        pad_ids = torch.full(
                            (1, pad_len),
                            self.tokenizer.pad_token_id,
                            device=self.device,
                        )
                        input_ids_i = torch.cat([pad_ids, ctx_ids, question_ids], dim=1)
                        padding_mask = torch.zeros_like(pad_ids)
                        padded_context_mask = torch.ones_like(ctx_ids)
                        context_ids_mask = torch.cat(
                            [padding_mask, padded_context_mask], dim=1
                        )
                        question_mask = torch.ones_like(question_ids)
                        mask_i = torch.cat([context_ids_mask, question_mask], dim=1)
                        batch_input_ids.append(input_ids_i)
                        batch_attn_mask.append(mask_i)

                    batched_inputs = torch.cat(
                        batch_input_ids, dim=0
                    )  # (B, max_ctx+q_len)
                    batched_attn = torch.cat(batch_attn_mask, dim=0)

                    cache = DynamicCache()
                    for layer_idx, (k, v) in enumerate(
                        zip(batched_gpu_keys, batched_gpu_values)
                    ):
                        cache.update(k.clone(), v.clone(), layer_idx)

                else:
                    # Ragged path: collect items that have a j-th question, tokenize each
                    # separately, right-pad questions to max_q_len (attention_mask = 0).
                    valid_bs = [b for b in range(B) if j < len(batch_qs[b])]
                    per_item_questions = [batch_qs[b][j] for b in valid_bs]
                    per_item_q_ids = []
                    for b, question in zip(valid_bs, per_item_questions):
                        full_question = context_prefix + " " + question
                        ctx_len = context_lens[b]
                        if self.tokenizer.chat_template is None:
                            question_suffix = "\n"
                        else:
                            separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                            template_context = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "user",
                                        "content": "Example of context" + separator,
                                    }
                                ],
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                            _, question_suffix = template_context.split(separator)
                        complete_question = (
                            full_question + question_suffix + answer_prefix
                        )
                        q_ids = self.tokenizer.encode(
                            complete_question,
                            return_tensors="pt",
                            add_special_tokens=False,
                        ).to(first_device)
                        per_item_q_ids.append(q_ids)

                    max_q_len = max(q.shape[1] for q in per_item_q_ids)
                    logger.info(
                        f"[join] j={j+1}/{batch_max_j} | {len(valid_bs)}/{B} items | "
                        f"sample='{per_item_questions[0][:100]}'"
                    )

                    batch_input_ids = []
                    batch_attn_mask = []
                    for vi, b in enumerate(valid_bs):
                        ctx_len = context_lens[b]
                        ctx_pad_len = max_ctx - ctx_len
                        q_ids = per_item_q_ids[vi]
                        q_pad_len = max_q_len - q_ids.shape[1]
                        ctx_ids = torch.full(
                            (1, ctx_len),
                            self.tokenizer.pad_token_id + 1,
                            device=self.device,
                        )
                        pad_ids = torch.full(
                            (1, ctx_pad_len),
                            self.tokenizer.pad_token_id,
                            device=self.device,
                        )
                        q_pad_ids = torch.full(
                            (1, q_pad_len),
                            self.tokenizer.pad_token_id,
                            device=self.device,
                        )
                        input_ids_i = torch.cat(
                            [pad_ids, ctx_ids, q_pad_ids, q_ids], dim=1
                        )
                        padding_mask = torch.zeros_like(pad_ids)
                        ctx_mask = torch.ones_like(ctx_ids)
                        context_ids_mask = torch.cat([padding_mask, ctx_mask], dim=1)
                        question_mask = torch.ones_like(q_ids)
                        q_pad_mask = torch.zeros_like(q_pad_ids)
                        mask_i = torch.cat(
                            [context_ids_mask, q_pad_mask, question_mask], dim=1
                        )
                        batch_input_ids.append(input_ids_i)
                        batch_attn_mask.append(mask_i)

                    batched_inputs = torch.cat(
                        batch_input_ids, dim=0
                    )  # (V, max_ctx+max_q_len)
                    batched_attn = torch.cat(batch_attn_mask, dim=0)

                    cache = DynamicCache()
                    for layer_idx, (k, v) in enumerate(
                        zip(batched_gpu_keys, batched_gpu_values)
                    ):
                        cache.update(
                            k[valid_bs].clone(), v[valid_bs].clone(), layer_idx
                        )

                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        input_ids=batched_inputs,
                        attention_mask=batched_attn,
                        past_key_values=cache,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                logits = generated.scores[0]  # (V, vocab)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                log_odds_batch = (log_probs[:, id1] - log_probs[:, id0]).cpu()  # (V,)
                decoded_batch = self.tokenizer.batch_decode(
                    generated.sequences[:, batched_inputs.shape[1] :],
                    skip_special_tokens=True,
                )

                for vi, b in enumerate(valid_bs):
                    answer = decoded_batch[vi]
                    lo = log_odds_batch[vi].item()
                    answers_per_text[batch_start + b][j] = answer
                    log_odds_per_text[batch_start + b][j] = lo

                del cache, batched_inputs, batched_attn, generated, logits, log_probs
                torch.cuda.empty_cache()

            logger.info(
                f"[join] Batch {batch_start // batch_size + 1} done: "
                f"{B} contexts × {batch_max_j} questions in {time.time() - _t_questions_start:.2f}s"
            )
            del batched_gpu_keys, batched_gpu_values

        return {
            "answers_per_text": answers_per_text,
            "log_odds_per_text": log_odds_per_text,
        }

    def compute_text_qa_ape_join_response(
        self,
        column_name: str,
        left_texts: List[str],
        right_texts: List[str],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
        temperature: float = 0.9,
        scale: float = 0.9,
        max_batch_size: int = 8,
    ) -> dict:
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        return asyncio.run(
            self._run_kv_cache_text_ape_join(
                column_name=column_name,
                left_texts=left_texts,
                right_texts=right_texts,
                question=question,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
                temperature=temperature,
                scale=scale,
                max_batch_size=max_batch_size,
            )
        )

    async def _run_kv_cache_text_ape_join(
        self,
        column_name: str,
        left_texts: List[str],
        right_texts: List[str],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
        temperature: float = 0.9,
        scale: float = 0.9,
        max_batch_size: int = 8,
    ) -> dict:
        """APE cartesian-product join: evaluate question over all N×M (left, right) pairs.

        Each pair's two KV caches are concatenated into a 4-tuple (key, val, pos, 2) per
        layer — exactly as in demo_ape_kvcache.py. Pairs are grouped by total KV length so
        no padding is ever needed (flash_attn causal=False has no padding-mask support).
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        model = self.pipe.model
        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        N, M = len(left_texts), len(right_texts)
        first_device = next(model.parameters()).device
        layer_devices = [
            model.model.layers[i].self_attn.q_proj.weight.device
            for i in range(len(model.model.layers))
        ]

        # ── 1. Load all KV caches from disk ──────────────────────────────
        def load_kv(path):
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                return data["key_cache"], data["value_cache"]
            return data.key_cache, data.value_cache

        left_kv = [
            load_kv(f"{save_dir}/cache_entry_{self.hash_text(t)}.pt")
            for t in left_texts
        ]
        right_kv = [
            load_kv(f"{save_dir}/cache_entry_{self.hash_text(t)}.pt")
            for t in right_texts
        ]
        left_lens = [kv[0][0].shape[2] for kv in left_kv]
        right_lens = [kv[0][0].shape[2] for kv in right_kv]

        logger.info(
            f"[ape_join] N={N} left, M={M} right, {N*M} total pairs, "
            f"compression_ratio={compression_ratio}"
        )

        # ── 2. Build query token ids ─────────────────────────────────────
        if boolean_question:
            full_question = (
                "Answer '1' if BOTH the context above AND the following text satisfy the condition, "
                "'0' otherwise. Do not add any other comments. " + question
            )
        else:
            full_question = question

        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": full_question}],
            add_generation_prompt=True,
            tokenize=False,
            add_special_tokens=False,
        )
        query_ids = self.tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(first_device)  # (1, q_len)

        max_new_tokens = 4 if boolean_question else 64
        id0 = self.tokenizer.convert_tokens_to_ids("0")
        id1 = self.tokenizer.convert_tokens_to_ids("1")

        # ── 3. Patch model with APE (save & restore to avoid corrupting other methods) ──
        saved_forwards = {
            id(m): m.forward
            for m in model.modules()
            if isinstance(m, (LlamaModel, LlamaAttention))
        }
        enable_attention_prefill_prefix(self.model_name, model)
        for m in model.modules():
            if isinstance(m, LlamaAttention):
                m.len_prefix = 0
        enable_attention_prefill_query(self.model_name, model, temperature, scale)

        # ── 4. Pre-allocate results ───────────────────────────────────────
        answers = [[""] * M for _ in range(N)]
        log_odds = [[0.0] * M for _ in range(N)]

        try:
            num_layers = len(left_kv[0][0])

            # ── Move all KV caches to GPU once (avoid repeated CPU→GPU transfers) ──
            for kv_list in [left_kv, right_kv]:
                for idx in range(len(kv_list)):
                    key_cache, val_cache = kv_list[idx]
                    kv_list[idx] = (
                        [
                            key_cache[layer].to(layer_devices[layer])
                            for layer in range(num_layers)
                        ],
                        [
                            val_cache[layer].to(layer_devices[layer])
                            for layer in range(num_layers)
                        ],
                    )

            for i, j in tqdm(
                [(i, j) for i in range(N) for j in range(M)],
                desc="APE join: processing pairs",
            ):
                len_A = left_lens[i]
                len_B = right_lens[j]

                # Assemble APE past_key_values: one 4-tuple per layer, batch size=1.
                past_kv = []
                for layer in range(num_layers):
                    dev = layer_devices[layer]
                    key = torch.cat(
                        [left_kv[i][0][layer], right_kv[j][0][layer]], dim=2
                    )  # (1, heads, len_A+len_B, d) — already on correct device
                    val = torch.cat(
                        [left_kv[i][1][layer], right_kv[j][1][layer]], dim=2
                    )
                    pos = (
                        torch.cat([torch.arange(len_A), torch.arange(len_B)])
                        .unsqueeze(0)
                        .to(dev)
                    )  # (1, len_A+len_B)
                    past_kv.append((key, val, pos, 2))

                with torch.no_grad():
                    generated = model.generate(
                        input_ids=query_ids,
                        attention_mask=torch.ones_like(query_ids),
                        past_key_values=past_kv,
                        cache_implementation=None,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                logits = generated.scores[0]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                lo = (log_probs[0, id1] - log_probs[0, id0]).cpu().item()
                decoded = self.tokenizer.decode(
                    generated.sequences[0, query_ids.shape[1] :],
                    skip_special_tokens=True,
                )
                answers[i][j] = decoded
                log_odds[i][j] = lo
                logger.info(
                    f"[ape_join] ({i},{j}) answer='{decoded}' log_odds={lo:.4f}"
                )

                del past_kv, generated
                torch.cuda.empty_cache()

        finally:
            # ── 6. Restore original forward methods ──────────────────────
            for m in model.modules():
                if id(m) in saved_forwards:
                    m.forward = saved_forwards[id(m)]

        return {"answers": answers, "log_odds": log_odds}

    # def compute_text_qa_ape_batch_join_response(
    #     self,
    #     column_name: str,
    #     left_texts: List[str],
    #     right_texts: List[str],
    #     question: str,
    #     compression_ratio: float,
    #     boolean_question: bool,
    #     cache_dir: str,
    #     temperature: float = 0.9,
    #     scale: float = 0.9,
    #     max_batch_size: int = 8,
    # ) -> dict:
    #     assert (
    #         compression_ratio in self.compression_ratios
    #     ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
    #     return asyncio.run(
    #         self._run_kv_cache_text_ape_batch_join(
    #             column_name=column_name,
    #             left_texts=left_texts,
    #             right_texts=right_texts,
    #             question=question,
    #             compression_ratio=compression_ratio,
    #             boolean_question=boolean_question,
    #             cache_dir=cache_dir,
    #             temperature=temperature,
    #             scale=scale,
    #             max_batch_size=max_batch_size,
    #         )
    #     )

    # async def _run_kv_cache_text_ape_batch_join(
    #     self,
    #     column_name: str,
    #     left_texts: List[str],
    #     right_texts: List[str],
    #     question: str,
    #     compression_ratio: float,
    #     boolean_question: bool,
    #     cache_dir: str,
    #     temperature: float = 0.9,
    #     scale: float = 0.9,
    #     max_batch_size: int = 8,
    # ) -> dict:
    #     """Batched APE join using flash_attn_varlen_func.

    #     Restructured to match TextQa join's loop pattern: batches B left contexts,
    #     then iterates over each right context. Left KV caches are pre-stacked on GPU
    #     once per batch and reused across all M right texts (VRAM→VRAM concat only).
    #     """
    #     assert self.pipe is not None, "Model pipeline is not initialized."
    #     assert self.tokenizer is not None, "Tokenizer is not initialized."

    #     model = self.pipe.model
    #     save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
    #     N, M = len(left_texts), len(right_texts)
    #     first_device = next(model.parameters()).device
    #     layer_devices = [
    #         model.model.layers[i].self_attn.q_proj.weight.device
    #         for i in range(len(model.model.layers))
    #     ]

    #     # ── 1. Load all KV caches from disk ──────────────────────────────
    #     def load_kv(path):
    #         data = torch.load(path, map_location="cpu", weights_only=False)
    #         if isinstance(data, dict):
    #             return data["key_cache"], data["value_cache"]
    #         return data.key_cache, data.value_cache

    #     left_kv = [
    #         load_kv(f"{save_dir}/cache_entry_{self.hash_text(t)}.pt")
    #         for t in left_texts
    #     ]
    #     right_kv = [
    #         load_kv(f"{save_dir}/cache_entry_{self.hash_text(t)}.pt")
    #         for t in right_texts
    #     ]
    #     left_lens = [kv[0][0].shape[2] for kv in left_kv]
    #     right_lens = [kv[0][0].shape[2] for kv in right_kv]

    #     # ── Auto batch size: based on left caches (like TextQa), halved for right concat overhead ──
    #     left_cache_paths = [
    #         f"{save_dir}/cache_entry_{self.hash_text(t)}.pt" for t in left_texts
    #     ]
    #     batch_size = self._get_max_batch_size(
    #         column_name=column_name,
    #         batch_size=None,  # force auto-calculation
    #         compression_ratio=compression_ratio,
    #         device_id=self.device_id,
    #         cache_dir=cache_dir,
    #         file_paths=left_cache_paths,
    #     )
    #     batch_size = max(
    #         batch_size // 2, 1
    #     )  # halve: each forward pass also holds 1 right cache

    #     logger.info(
    #         f"[ape_batch_join] N={N} left, M={M} right, {N*M} total pairs, "
    #         f"compression_ratio={compression_ratio}, batch_size={batch_size}"
    #     )

    #     # ── 2. Build query token ids ─────────────────────────────────────
    #     if boolean_question:
    #         full_question = (
    #             "Answer '1' if BOTH the context above AND the following text satisfy the condition, "
    #             "'0' otherwise. Do not add any other comments. " + question
    #         )
    #     else:
    #         full_question = question

    #     formatted = self.tokenizer.apply_chat_template(
    #         [{"role": "user", "content": full_question}],
    #         add_generation_prompt=True,
    #         tokenize=False,
    #         add_special_tokens=False,
    #     )
    #     query_ids = self.tokenizer(
    #         formatted, return_tensors="pt", add_special_tokens=False
    #     ).input_ids.to(first_device)  # (1, q_len)

    #     max_new_tokens = 4 if boolean_question else 64
    #     id0 = self.tokenizer.convert_tokens_to_ids("0")
    #     id1 = self.tokenizer.convert_tokens_to_ids("1")

    #     # ── 3. Patch model with varlen APE ───────────────────────────────
    #     saved_forwards = {
    #         id(m): m.forward
    #         for m in model.modules()
    #         if isinstance(m, (LlamaModel, LlamaAttention))
    #     }
    #     enable_attention_prefill_prefix(self.model_name, model)
    #     for m in model.modules():
    #         if isinstance(m, LlamaAttention):
    #             m.len_prefix = 0
    #     enable_llama_attention_prefill_query_varlen(model, temperature, scale)

    #     # ── 4. Pre-allocate results ───────────────────────────────────────
    #     answers = [[""] * M for _ in range(N)]
    #     log_odds = [[0.0] * M for _ in range(N)]

    #     try:
    #         num_layers = len(left_kv[0][0])

    #         # ── Move all KV caches to GPU once (avoid repeated CPU→GPU transfers) ──
    #         for kv_list in [left_kv, right_kv]:
    #             for idx in range(len(kv_list)):
    #                 key_cache, val_cache = kv_list[idx]
    #                 kv_list[idx] = (
    #                     [
    #                         key_cache[layer].to(layer_devices[layer])
    #                         for layer in range(num_layers)
    #                     ],
    #                     [
    #                         val_cache[layer].to(layer_devices[layer])
    #                         for layer in range(num_layers)
    #                     ],
    #                 )

    #         # ── 5. Process batches of left contexts × all right texts ─────
    #         # Mirrors TextQa join: stack B left caches once, reuse for all M right texts.
    #         for left_batch_start in tqdm(
    #             range(0, N, batch_size),
    #             desc="APE batch join: processing batches of left contexts",
    #         ):
    #             batch_left_indices = list(
    #                 range(left_batch_start, min(left_batch_start + batch_size, N))
    #             )
    #             B = len(batch_left_indices)
    #             left_context_lens = [left_lens[i] for i in batch_left_indices]
    #             max_left = max(left_context_lens)

    #             # ── Step A: Pre-pad and stack B left KV caches on GPU (done ONCE per batch) ──
    #             batched_left_keys = []  # per-layer: (B, heads, max_left, d)
    #             batched_left_vals = []
    #             for layer in range(num_layers):
    #                 keys_padded, vals_padded = [], []
    #                 for b, i in enumerate(batch_left_indices):
    #                     k = left_kv[i][0][layer]
    #                     v = left_kv[i][1][layer]
    #                     pad_len = max_left - left_lens[i]
    #                     if pad_len > 0:
    #                         k = torch.nn.functional.pad(
    #                             k, (0, 0, pad_len, 0)
    #                         ).contiguous()
    #                         v = torch.nn.functional.pad(
    #                             v, (0, 0, pad_len, 0)
    #                         ).contiguous()
    #                     keys_padded.append(k)
    #                     vals_padded.append(v)
    #                 batched_left_keys.append(torch.cat(keys_padded, dim=0))
    #                 batched_left_vals.append(torch.cat(vals_padded, dim=0))

    #             # ── Step B: For each right text, concat with pre-stacked left and run forward pass ──
    #             _t_questions_start = time.time()
    #             for j in range(M):
    #                 right_len = right_lens[j]

    #                 # Offsets and context lengths for varlen attention
    #                 b_len_contexts = [
    #                     left_context_lens[b] + right_len for b in range(B)
    #                 ]
    #                 # total_len = max_left + right_len for all items (left is pre-padded to max_left)
    #                 b_offsets = [max_left - left_context_lens[b] for b in range(B)]

    #                 # Build position tensor once (same for all layers)
    #                 pos_parts = []
    #                 for b in range(B):
    #                     pad_len = b_offsets[b]
    #                     p = torch.cat(
    #                         [
    #                             torch.zeros(pad_len, dtype=torch.long),
    #                             torch.arange(left_context_lens[b]),
    #                             torch.arange(right_len),
    #                         ]
    #                     )
    #                     pos_parts.append(p.unsqueeze(0))
    #                 batched_pos_cpu = torch.cat(
    #                     pos_parts, dim=0
    #                 )  # (B, max_left + right_len)

    #                 # Build batched past_key_values: concat pre-stacked left + expanded right[j]
    #                 past_kv = []
    #                 for layer in range(num_layers):
    #                     dev = layer_devices[layer]
    #                     # Expand right[j] to B copies and concatenate (VRAM→VRAM, expand is a view)
    #                     right_k = right_kv[j][0][layer].expand(
    #                         B, -1, -1, -1
    #                     )  # (B, heads, right_len, d)
    #                     right_v = right_kv[j][1][layer].expand(B, -1, -1, -1)

    #                     batched_key = torch.cat(
    #                         [batched_left_keys[layer], right_k], dim=2
    #                     )
    #                     batched_val = torch.cat(
    #                         [batched_left_vals[layer], right_v], dim=2
    #                     )

    #                     past_kv.append(
    #                         (
    #                             batched_key,
    #                             batched_val,
    #                             batched_pos_cpu.to(dev),
    #                             (b_offsets, b_len_contexts),
    #                         )
    #                     )

    #                 # Batched query
    #                 batched_query = query_ids.expand(B, -1)  # (B, q_len)

    #                 with torch.no_grad():
    #                     generated = model.generate(
    #                         input_ids=batched_query,
    #                         attention_mask=torch.ones_like(batched_query),
    #                         past_key_values=past_kv,
    #                         cache_implementation=None,
    #                         max_new_tokens=max_new_tokens,
    #                         num_beams=1,
    #                         do_sample=False,
    #                         temperature=1.0,
    #                         output_scores=True,
    #                         return_dict_in_generate=True,
    #                         pad_token_id=self.tokenizer.eos_token_id,
    #                     )

    #                 # Extract per-item results
    #                 for b, i in enumerate(batch_left_indices):
    #                     logits = generated.scores[0][b]
    #                     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    #                     lo = (log_probs[id1] - log_probs[id0]).cpu().item()
    #                     decoded = self.tokenizer.decode(
    #                         generated.sequences[b, query_ids.shape[1] :],
    #                         skip_special_tokens=True,
    #                     )
    #                     answers[i][j] = decoded
    #                     log_odds[i][j] = lo
    #                     logger.info(
    #                         f"[ape_batch_join] ({i},{j}) answer='{decoded}' log_odds={lo:.4f}"
    #                     )

    #                 del past_kv, generated
    #                 torch.cuda.empty_cache()

    #             logger.info(
    #                 f"[ape_batch_join] Left batch {left_batch_start // batch_size + 1} done: "
    #                 f"{B} left contexts × {M} right texts in {time.time() - _t_questions_start:.2f}s"
    #             )
    #             del batched_left_keys, batched_left_vals
    #             torch.cuda.empty_cache()

    #     finally:
    #         # ── 6. Restore original forward methods ──────────────────────
    #         for m in model.modules():
    #             if id(m) in saved_forwards:
    #                 m.forward = saved_forwards[id(m)]

    #     return {"answers": answers, "log_odds": log_odds}

    def compute_hidden_state_response(
        self,
        column_name: str,
        texts: List[str],
        suffix_prompt: str,
        compression_ratio: float,
        cache_dir: str,
    ) -> dict:
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        return asyncio.run(
            self._run_hidden_state(
                column_name=column_name,
                texts=texts,
                suffix_prompt=suffix_prompt,
                compression_ratio=compression_ratio,
                cache_dir=cache_dir,
            )
        )

    async def _run_hidden_state(
        self,
        column_name: str,
        texts: List[str],
        suffix_prompt: str,
        compression_ratio: float,
        cache_dir: str,
    ) -> dict:
        """Prefill-only forward pass: load KV cache, append suffix_prompt tokens, extract
        the last-layer hidden state of the final token. No generation.

        Returns {"hidden_states": [[float, ...], ...]} — one float vector per text.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        first_device = next(self.pipe.model.parameters()).device
        layer_devices = [
            self.pipe.model.model.layers[i].self_attn.q_proj.weight.device
            for i in range(len(self.pipe.model.model.layers))
        ]

        # Tokenize suffix prompt using the same chat-template logic as _run_kv_cache_text.
        # We use a dummy 1-token context to get the suffix portion of the template.
        if self.tokenizer.chat_template is None:
            suffix_str = suffix_prompt + "\n"
        else:
            separator = "\n#"
            template_context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "X" + separator + suffix_prompt}],
                add_generation_prompt=False,
                tokenize=False,
            )
            _, suffix_str = template_context.split(separator, 1)

        suffix_ids = self.tokenizer.encode(
            suffix_str, return_tensors="pt", add_special_tokens=False
        ).to(first_device)  # (1, q_len)

        cache_files = [f"{save_dir}/cache_entry_{self.hash_text(t)}.pt" for t in texts]
        for cf in cache_files:
            assert os.path.exists(cf), f"Cache file not found: {cf}"

        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        first_cache_data = torch.load(
            cache_files[0], map_location="cpu", weights_only=False
        )
        if isinstance(first_cache_data, dict):
            _ctx_tokens = first_cache_data["key_cache"][0].shape[2]
        else:
            _ctx_tokens = first_cache_data.key_cache[0].shape[2]
        del first_cache_data
        _q_tokens = suffix_ids.shape[1]
        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            layer_devices=self._unique_layer_devices(),
            cache_dir=cache_dir,
            file_paths=cache_files,
            max_question_tokens=_q_tokens,
            max_context_tokens=_ctx_tokens,
        )

        suffix_len = suffix_ids.shape[1]
        all_hidden_states = []

        for batch_start in tqdm(
            range(0, len(texts), batch_size),
            desc="HiddenState: extracting conditioned embeddings",
        ):
            batch_cache_files = cache_files[batch_start : batch_start + batch_size]
            batch_size_actual = len(batch_cache_files)

            # 1. Load KV caches
            caches = []
            context_lengths = []
            for cf in batch_cache_files:
                cache_data = torch.load(cf, map_location="cpu", weights_only=False)
                if isinstance(cache_data, dict):
                    key_cache = cache_data["key_cache"]
                    value_cache = cache_data["value_cache"]
                else:
                    key_cache = cache_data.key_cache
                    value_cache = cache_data.value_cache
                key_cache = [k.to(layer_devices[j]) for j, k in enumerate(key_cache)]
                value_cache = [
                    v.to(layer_devices[j]) for j, v in enumerate(value_cache)
                ]
                caches.append((key_cache, value_cache))
                context_lengths.append(key_cache[0].shape[2])

            max_context_len = max(context_lengths)

            # 2. Build padded input_ids and attention masks
            #    Left-pad shorter contexts; suffix is same for all (no question padding needed)
            batch_input_ids = []
            batch_attention_masks = []
            for ctx_len in context_lengths:
                pad_len = max_context_len - ctx_len
                ctx_placeholder = torch.full(
                    (1, ctx_len), self.tokenizer.pad_token_id + 1, device=first_device
                )
                if pad_len > 0:
                    padding = torch.full(
                        (1, pad_len), self.tokenizer.pad_token_id, device=first_device
                    )
                    padded_ctx = torch.cat([padding, ctx_placeholder], dim=1)
                    ctx_mask = torch.cat(
                        [
                            torch.zeros(1, pad_len, device=first_device),
                            torch.ones(1, ctx_len, device=first_device),
                        ],
                        dim=1,
                    )
                else:
                    padded_ctx = ctx_placeholder
                    ctx_mask = torch.ones(1, ctx_len, device=first_device)

                suffix_mask = torch.ones(1, suffix_len, device=first_device)
                batch_input_ids.append(torch.cat([padded_ctx, suffix_ids], dim=1))
                batch_attention_masks.append(torch.cat([ctx_mask, suffix_mask], dim=1))

            batched_inputs = torch.cat(
                batch_input_ids, dim=0
            )  # (B, max_ctx + suffix_len)
            batched_attention_mask = torch.cat(batch_attention_masks, dim=0)

            # 3. Build batched DynamicCache (same pattern as _run_kv_cache_text)
            batched_cache_layers = []
            for layer_kv_pairs in zip(*[zip(kc, vc) for kc, vc in caches]):
                max_seq_len = max(k.shape[2] for k, _ in layer_kv_pairs)
                keys_padded = []
                values_padded = []
                for k, v in layer_kv_pairs:
                    pad_len = max_seq_len - k.shape[2]
                    k_p = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else k
                    )
                    v_p = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else v
                    )
                    keys_padded.append(k_p)
                    values_padded.append(v_p)
                batched_cache_layers.append(
                    (torch.cat(keys_padded, dim=0), torch.cat(values_padded, dim=0))
                )

            padded_cache = DynamicCache()
            for layer_idx, (keys, values) in enumerate(batched_cache_layers):
                padded_cache.update(keys, values, layer_idx)

            # 4. Prefill-only forward pass (no generation)
            with torch.no_grad():
                outputs = self.pipe.model.model(
                    input_ids=batched_inputs,
                    attention_mask=batched_attention_mask,
                    past_key_values=padded_cache,
                    use_cache=False,
                    output_hidden_states=True,
                )

            # 5. Last-layer hidden state of the last token for each batch item.
            #    With left-padded context, the last real token is always at position [:, -1, :].
            last_hidden = (
                outputs.hidden_states[-1][:, -1, :].float().cpu()
            )  # (B, hidden_size)
            for vec in last_hidden:
                all_hidden_states.append(vec.tolist())

            del padded_cache, outputs, batched_cache_layers
            torch.cuda.empty_cache()

        return {"hidden_states": all_hidden_states}

    async def _run_hidden_state_similarity(
        self,
        left_column_name: str,
        right_column_name: str,
        left_texts: List[str],
        right_texts: List[str],
        pair_left_indices: List[int],
        pair_right_indices: List[int],
        suffix_prompt: str,
        compression_ratio: float,
        cache_dir: str,
    ) -> dict:
        """Compute cosine similarities between paired hidden states on GPU.

        Processes left and right texts separately so the caller can attribute
        runtime to each side independently (matching ExtractAndMatchFilter's
        runtime1/runtime2 reporting convention).
        Returns {"similarities": [float, ...], "runtime_left": float, "runtime_right": float}.
        """
        import time as _time

        # 1. Hidden states for left texts
        t0 = _time.time()
        left_result = await self._run_hidden_state(
            column_name=left_column_name,
            texts=left_texts,
            suffix_prompt=suffix_prompt,
            compression_ratio=compression_ratio,
            cache_dir=cache_dir,
        )
        runtime_left = _time.time() - t0
        torch.cuda.empty_cache()

        # 2. Hidden states for right texts
        t1 = _time.time()
        right_result = await self._run_hidden_state(
            column_name=right_column_name,
            texts=right_texts,
            suffix_prompt=suffix_prompt,
            compression_ratio=compression_ratio,
            cache_dir=cache_dir,
        )
        runtime_right = _time.time() - t1

        # 3. GPU cosine similarity via matmul on unique vectors — avoids expanding to pair-space
        #    (pair-expansion would be NP × hidden_size = 16 GB for 1M pairs × 4096-dim)
        t2 = _time.time()
        first_device = next(self.pipe.model.parameters()).device
        left_tensor = torch.tensor(
            left_result["hidden_states"], dtype=torch.float32, device=first_device
        )
        right_tensor = torch.tensor(
            right_result["hidden_states"], dtype=torch.float32, device=first_device
        )
        left_norm = torch.nn.functional.normalize(
            left_tensor, dim=1
        )  # (NL, hidden_size)
        right_norm = torch.nn.functional.normalize(
            right_tensor, dim=1
        )  # (NR, hidden_size)
        sim_matrix = left_norm @ right_norm.T  # (NL, NR)
        sims = sim_matrix[pair_left_indices, pair_right_indices].cpu().tolist()
        runtime_sim = _time.time() - t2
        return {
            "similarities": sims,
            "runtime_left": runtime_left,
            "runtime_right": runtime_right,
            "runtime_sim": runtime_sim,
        }

    def compute_hidden_state_similarity_response(
        self,
        left_column_name: str,
        right_column_name: str,
        left_texts: List[str],
        right_texts: List[str],
        pair_left_indices: List[int],
        pair_right_indices: List[int],
        suffix_prompt: str,
        compression_ratio: float,
        cache_dir: str,
    ) -> dict:
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        return asyncio.run(
            self._run_hidden_state_similarity(
                left_column_name=left_column_name,
                right_column_name=right_column_name,
                left_texts=left_texts,
                right_texts=right_texts,
                pair_left_indices=pair_left_indices,
                pair_right_indices=pair_right_indices,
                suffix_prompt=suffix_prompt,
                compression_ratio=compression_ratio,
                cache_dir=cache_dir,
            )
        )

    async def _generate_cache_for_text(
        self, text_content, cache_filename, compression_ratio
    ):
        """Generate and save text KV cache for a single text."""
        assert (
            self.pipe is not None
        ), "Model pipeline for cache generation is not initialized."

        try:
            context = text_content[
                : min(128000, len(text_content))
            ]  # Limit context length
            answer_prefix = "Answer: "

            # Generate cache using preprocess method
            inputs = self.pipe.preprocess(
                context=context,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
            )

            context_ids = inputs["context_ids"]
            cache = DynamicCache()

            press = self.presses[compression_ratio]
            with torch.inference_mode():
                with (
                    press(self.pipe.model)
                    if press is not None
                    else contextlib.nullcontext()
                ):
                    # Run the model without the lm head for pre-filling
                    self.pipe.model.model(
                        input_ids=context_ids,
                        past_key_values=cache,
                        use_cache=True,
                        output_attentions=self.pipe.output_attentions(press),
                    )

            # Move each cache layer to the same device as the corresponding model layer
            # This is necessary when using device_map="auto" which can distribute layers across GPUs
            # for layer_idx in range(len(cache.key_cache)):
            #    layer_device = self.pipe.model.model.layers[layer_idx].self_attn.q_proj.weight.device
            #    cache.key_cache[layer_idx] = cache.key_cache[layer_idx].to(layer_device)
            #    cache.value_cache[layer_idx] = cache.value_cache[layer_idx].to(layer_device)

            # Save cache to disk as a plain dict to avoid DynamicCache API version issues
            cache_data = {
                "key_cache": [layer.keys.detach().cpu() for layer in cache.layers],
                "value_cache": [layer.values.detach().cpu() for layer in cache.layers],
            }

            logger.info(f"Saving cache to: {cache_filename}")

            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            torch.save(cache_data, cache_filename)

            if not os.path.exists(cache_filename):
                logger.error(f"Cache file was not actually created: {cache_filename}")

            # Cleanup
            del cache
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating cache for text {cache_filename}: {str(e)}")
            raise


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "compression_ratios": list(model_wrapper.compression_ratios),
        }, 200


class PrepareCaches(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "texts": [
                "text1",
                "text2",
            ],
            "cache_dir": "/path/to/cache/dir"
            "compression_ratio": 0.5
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        cache_dir = data["cache_dir"]
        compresion_ratio = data["compression_ratio"]
        assert model_wrapper is not None
        asyncio.run(
            model_wrapper.prepare_caches(
                column_name=column_name,
                texts=texts,
                cache_dir=cache_dir,
                compression_ratio=compresion_ratio,
            )
        )
        return {"status": "cache_ready"}, 200


class TextQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "texts": [
                "text 1",
                "text 2",
            ],
            "questions": [
                "How many cats are there?"
                "How many dogs are there?"
            ],
            "compression_ratio": 0.5,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        questions = data["questions"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        assert (
            len(texts) == len(questions)
        ), f"Number of texts {len(texts)} must match number of questions {len(questions)}"
        responses = model_wrapper.compute_text_qa_response(
            column_name=column_name,
            texts=texts,
            questions=questions,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


class TextQAJoin(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "column_name": "reviews.reviewtext",
            "unique_texts": ["text1", "text2", ...],
            "questions_per_text": [["q1_for_text1", "q2_for_text1"], ["q1_for_text2"], ...],
            "compression_ratio": 0.0,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir"
        }
        Returns answers_per_text and log_odds_per_text with the same nested structure.
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        unique_texts = data["unique_texts"]
        questions_per_text = data["questions_per_text"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        assert (
            len(unique_texts) == len(questions_per_text)
        ), f"unique_texts ({len(unique_texts)}) must match questions_per_text ({len(questions_per_text)})"
        responses = model_wrapper.compute_text_qa_join_response(
            column_name=column_name,
            unique_texts=unique_texts,
            questions_per_text=questions_per_text,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


class TextQAApeJoin(Resource):
    def post(self):
        """APE cartesian-product join over all N×M (left, right) text pairs.

        Expects JSON of the form:
        {
            "column_name":      "reviews.reviewtext",
            "left_texts":       ["text1", ...],
            "right_texts":      ["text2", ...],
            "question":         "Do they both express positive sentiment?",
            "compression_ratio": 0.0,
            "boolean":          true,
            "cache_dir":        "/path/to/cache/dir",
            "temperature":      0.9,
            "scale":            0.9,
            "max_batch_size":   8
        }
        Returns {"answers": N×M list of strings, "log_odds": N×M list of floats}
        """
        data = request.get_json(force=True)
        assert model_wrapper is not None
        try:
            responses = model_wrapper.compute_text_qa_ape_join_response(
                column_name=data["column_name"],
                left_texts=data["left_texts"],
                right_texts=data["right_texts"],
                question=data["question"],
                compression_ratio=data["compression_ratio"],
                boolean_question=data["boolean"],
                cache_dir=data["cache_dir"],
                temperature=data.get("temperature", 0.9),
                scale=data.get("scale", 0.9),
                max_batch_size=data.get("max_batch_size", 8),
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.error(f"[text_qa_ape_join] Exception:\n{tb}")
            return {"error": str(e), "traceback": tb}, 500
        return responses, 200


# class TextQAApeBatchJoin(Resource):
#     def post(self):
#         """Batched APE join using flash_attn_varlen_func.
#
#         Same API as TextQAApeJoin but processes multiple pairs per forward pass.
#         """
#         data = request.get_json(force=True)
#         assert model_wrapper is not None
#         try:
#             responses = model_wrapper.compute_text_qa_ape_batch_join_response(
#                 column_name=data["column_name"],
#                 left_texts=data["left_texts"],
#                 right_texts=data["right_texts"],
#                 question=data["question"],
#                 compression_ratio=data["compression_ratio"],
#                 boolean_question=data["boolean"],
#                 cache_dir=data["cache_dir"],
#                 temperature=data.get("temperature", 0.9),
#                 scale=data.get("scale", 0.9),
#                 max_batch_size=data.get("max_batch_size", 8),
#             )
#         except Exception as e:
#             import traceback
#
#             tb = traceback.format_exc()
#             logger.error(f"[text_qa_ape_batch_join] Exception:\n{tb}")
#             return {"error": str(e), "traceback": tb}, 500
#         return responses, 200


class TextQADirect(Resource):
    def post(self):
        """Direct QA — no KV caches required.

        Expects JSON of the form:
        {
            "questions": ["has the same sentiment as {right_text}?", ...],
            "contexts": ["{left_text}", ...],
            "boolean": true
        }
        Returns {"answers": [...], "log_odds": [...]}
        """
        data = request.get_json(force=True)
        questions = data["questions"]
        contexts = data["contexts"]
        boolean_question = data["boolean"]
        batch_size = data.get("batch_size", 128)
        assert model_wrapper is not None
        responses = model_wrapper.compute_direct_qa_response(
            questions=questions,
            contexts=contexts,
            boolean_question=boolean_question,
            batch_size=batch_size,
        )
        responses["telemetry"]["requested_cache_policy"] = data.get(
            "cache_policy", "caller_managed"
        )
        return responses, 200


class Generate(Resource):
    def post(self):
        data = request.get_json(force=True)
        assert model_wrapper is not None
        return model_wrapper.compute_generation_response(
            data["messages"], int(data.get("max_new_tokens", 512))
        ), 200


class HiddenState(Resource):
    def post(self):
        """Prefill-only hidden-state extraction for similarity blocking.

        Expects JSON:
        {
            "column_name": "reviews.reviewtext",
            "texts": ["text1", ...],
            "suffix_prompt": "Extract the movie title",
            "compression_ratio": 0.0,
            "cache_dir": "/path/to/cache/dir"
        }
        Returns {"hidden_states": [[float, ...], ...]}
        """
        data = request.get_json(force=True)
        assert model_wrapper is not None
        try:
            responses = model_wrapper.compute_hidden_state_response(
                column_name=data["column_name"],
                texts=data["texts"],
                suffix_prompt=data["suffix_prompt"],
                compression_ratio=data["compression_ratio"],
                cache_dir=data["cache_dir"],
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.error(f"[hidden_state] Exception:\n{tb}")
            return {"error": str(e), "traceback": tb}, 500
        return responses, 200


class HiddenStateSimilarity(Resource):
    def post(self):
        """GPU cosine similarity between paired KV-cache conditioned hidden states.

        Expects JSON:
        {
            "left_column_name": "reviews.reviewtext",
            "right_column_name": "reviews.reviewtext",
            "left_texts": ["text1", "text2"],       // NL unique left texts
            "right_texts": ["text2", "text3"],       // NR unique right texts
            "pair_left_indices": [0, 0, 1, ...],    // NP indices into left_texts
            "pair_right_indices": [0, 1, 0, ...],   // NP indices into right_texts
            "suffix_prompt": "Extract the movie title",
            "compression_ratio": 0.0,
            "cache_dir": "/path/to/cache"
        }
        Returns {"similarities": [float, ...]}  -- one float per pair
        """
        data = request.get_json(force=True)
        assert model_wrapper is not None
        try:
            responses = model_wrapper.compute_hidden_state_similarity_response(
                left_column_name=data["left_column_name"],
                right_column_name=data["right_column_name"],
                left_texts=data["left_texts"],
                right_texts=data["right_texts"],
                pair_left_indices=data["pair_left_indices"],
                pair_right_indices=data["pair_right_indices"],
                suffix_prompt=data["suffix_prompt"],
                compression_ratio=data["compression_ratio"],
                cache_dir=data["cache_dir"],
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.error(f"[hidden_state_similarity] Exception:\n{tb}")
            return {"error": str(e), "traceback": tb}, 500
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(TextQA, "/text_qa")
api.add_resource(TextQAJoin, "/text_qa_join")
api.add_resource(TextQAApeJoin, "/text_qa_ape_join")
# api.add_resource(TextQAApeBatchJoin, "/text_qa_ape_batch_join")
api.add_resource(TextQADirect, "/text_qa_direct")
api.add_resource(Generate, "/generate")
api.add_resource(HiddenState, "/hidden_state")
api.add_resource(HiddenStateSimilarity, "/hidden_state_similarity")
api.add_resource(PrepareCaches, "/prepare_caches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    parser.add_argument(
        "--gold-vanilla",
        action="store_true",
        help="Use vanilla inference (no KV caching) for CR=0 with 70B models",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=sorted(PORT_KV_TEXT_QA.keys()),
        default=MODEL_NAME,
        help="Name of the model to use",
    )
    parser.add_argument("--port", type=int, default=None, help="HTTP port override for isolated jobs.")
    args = parser.parse_args()
    device_id = args.device_id
    gold_vanilla = args.gold_vanilla
    model_name = args.model_name
    model_wrapper = KvTextQaModelWrapper(
        model_name,
        device_id,
        batch_sizes=[None, None, None, None, None, None, None],
        compression_ratios=[
            0.0,
            0.5,
            0.8,
            0.9,
            0.6,
            0.4,
            0.3,
        ],
        gold_vanilla=gold_vanilla,
    )
    app.run(host="127.0.0.1", port=args.port or PORT_KV_TEXT_QA.get(model_name), debug=False)
