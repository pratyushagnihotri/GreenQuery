import argparse
import json
import asyncio
import logging
from collections import defaultdict
from tqdm import tqdm
from flask import Flask, request
from flask_restful import Resource, Api
from typing import Dict, List, Tuple

from reasondb.backends.kv_cache_base import KVCachingBackendBase
from reasondb.backends.vision_model import PORT_KV_VISION
from PIL import Image, ImageFile
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress

from transformers import DynamicCache, pipeline  # type: ignore
import os
import contextlib

import base64
from io import BytesIO
from transformers import AutoProcessor

from reasondb.memory_footprint.memory_report import compute_memory_footprints
import numpy as np


def _iter_cache_layers(cache):
    """Yield (key_tensor, value_tensor) per layer for old and new DynamicCache."""
    if hasattr(cache, "layers"):
        # transformers 5.x
        for layer in cache.layers:
            yield layer.keys, layer.values
    elif hasattr(cache, "_cache") and cache._cache and hasattr(cache._cache[0], "key_states"):
        # transformers 4.50.3
        for item in cache._cache:
            yield item.key_states, item.value_states
    else:
        yield from zip(cache.__dict__["key_cache"], cache.__dict__["value_cache"])


MODEL_NAME = "llava-hf/llava-next-72b-hf"
IMAGE_MAX_PIXELS = 1400 * 1400
MODEL_TAG = {
    "llava-hf/llava-next-72b-hf": "llava-70B",
    "llava-hf/llama3-llava-next-8b-hf": "llava-8B",
}
PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
}
# Enable loading of truncated images
# Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


class KvImageQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id,
        compression_ratios=(0.0, 0.5, 0.8, 0.9, 0.99),
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

        self.model_tag = MODEL_TAG.get(model_name, "unknown_model")

        # Will be initialized in init()
        self.init()

    def compute_image_qa_response(
        self,
        column_name: str,
        image_paths: List[str],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ) -> Dict[str, Dict]:
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        responses, log_odds = asyncio.run(
            self._run_kv_cache_multimodal(
                column_name=column_name,
                image_paths=image_paths,
                question=question,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
            )
        )
        return {"answers": responses, "log_odds": log_odds}

    def compute_image_qa_direct_response(
        self,
        image_paths: List[str],
        question: str,
        boolean_question: bool,
    ) -> Dict[str, Dict]:
        """Run plain LLaVA image inference without any persistent KV-cache files."""
        assert self.pipe is not None and self.pipe.tokenizer is not None
        processor = AutoProcessor.from_pretrained(self.model_name)
        instruction = (
            "Answer the following question about the image with '1' or '0'. "
            "Do not add any other comments."
            if boolean_question
            else "Answer the following question about the image. Do not add any other comments."
        )
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": f"{instruction} {question}"}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + "Answer: "
        device = next(self.pipe.model.parameters()).device
        id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
        id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")
        answers, log_odds = {}, {}
        input_tokens = output_tokens = batch_count = 0
        batch_size = int(os.environ.get("GREENQUERY_VISION_QA_BATCH_SIZE", "8"))
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start : start + batch_size]
            images = [self.deserialize_image(path) for path in paths]
            for image in images:
                image.thumbnail((336, 336))
            inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
            input_len = inputs["input_ids"].shape[1]
            input_tokens += int(inputs["attention_mask"].sum().item())
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    **inputs,
                    max_new_tokens=4 if boolean_question else 64,
                    do_sample=False,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            logits = generated.scores[0]
            scores = torch.nn.functional.log_softmax(logits, dim=-1)
            decoded = processor.batch_decode(generated.sequences[:, input_len:], skip_special_tokens=True)
            output_tokens += int((generated.sequences[:, input_len:] != self.pipe.tokenizer.pad_token_id).sum().item())
            for path, answer, score in zip(paths, decoded, (scores[:, id1] - scores[:, id0]).tolist()):
                answers[path] = answer.strip()
                log_odds[path] = float(score)
            batch_count += 1
            del images, inputs, generated, logits, scores
            torch.cuda.empty_cache()
        return {
            "answers": answers,
            "log_odds": log_odds,
            "telemetry": {
                "model_name": self.model_name,
                "model_revision": getattr(self.pipe.model.config, "_commit_hash", None),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "inference_batch_count": batch_count,
                "gpu_index": self.device_id,
            },
        }

    def hash_path(self, path: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(path.encode()).hexdigest()

    async def prepare_caches(
        self,
        column_name: str,
        image_paths: List[str],
        cache_dir: str,
        compression_ratio: float,
    ):
        # Skip cache preparation if gold_vanilla is enabled for CR=0 and 72b model
        if self.gold_vanilla and compression_ratio == 0.0 and "72b" in self.model_name:
            logger.info(
                "Skipping cache preparation for gold-vanilla mode (CR=0, 72b model)"
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

        for i, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            hash_name = self.hash_path(image_path)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                image = self.deserialize_image(image_path)
                await self._generate_cache_for_image(
                    image, cache_filename, compression_ratio
                )
            except Exception as e:
                logger.warning(
                    f"Error processing image {i} {image_path} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)
        compute_memory_footprints(cache_dir, column_name, cache_filenames, model_name=self.model_name)

        with open(
            f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}/ERRORS.json",
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
        """Initialize the llava-next model pipeline and compression settings."""
        logger.info("Setting up KV Cache Filter with llava-next model...")

        # Set up device
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"

        # Set PIL image size limit
        Image.MAX_IMAGE_PIXELS = 250000000

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
                    f"Error initializing model with args {x}: {str(e)}", exc_info=True
                )
        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()

        # Set up compression press
        if self.press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{self.press_name}'. Available options: {list(PRESS.keys())}"
            )
        self.presses = {
            cr: KeyRerotationPress(PRESS[self.press_name](compression_ratio=cr))
            for cr in self.compression_ratios
        }

        logger.info(
            f"Using press {self.press_name} with compression_ratios={self.compression_ratios}",
        )
        logger.info(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

    def _get_llm_layers(self):
        """Return the transformer layers of the LLM part, compatible with old and new transformers APIs.

        Old transformers: LlavaNextForConditionalGeneration.language_model = LlamaModel → .layers
        New transformers: LlavaNextForConditionalGeneration.model.language_model = LlamaModel → .layers
        """
        model = self.pipe.model
        if hasattr(model, "language_model"):
            return model.language_model.layers
        return model.model.language_model.layers

    def _get_max_batch_size_image_join(
        self,
        file_paths: List[str],
        max_right_tokens: int,
        max_context_tokens: int,
    ) -> int:
        """Compute the max safe batch size for image join.

        The generic text-join formula only accounts for KV-cache file sizes and
        ignores two additional costs that dominate for images:
          - ViT activation memory:  Q/K/V tensors per layer through the vision encoder
          - LLM activation memory:  Q/K/V + FFN tensors per layer over the full
                                    (max_ctx + right_len) input sequence

        With flash-attention, attention weights are never fully materialised, so
        only the three projection tensors and the FFN intermediate need to be counted.

        Parameters
        ----------
        file_paths         : paths to all left-side KV-cache .pt files
        max_right_tokens   : token count of one right-side input (right_ids length)
        max_context_tokens : token count of one left KV cache
        """
        import math
        from reasondb.memory_footprint.memory_report import get_biggest_file_size_gb

        save_dir = os.path.dirname(file_paths[0])
        filenames = [os.path.basename(f) for f in file_paths]
        kv_size_gb = get_biggest_file_size_gb(save_dir, filenames)

        # (1) KV cache loading overhead: 3× disk size (same as text-join formula)
        kv_mem_gb = 3 * kv_size_gb

        # (2) Right-side token activations (same q_len/ctx_len scaling as text join)
        q_mem_gb = (
            kv_size_gb * (max_right_tokens / max_context_tokens)
            if max_context_tokens > 0
            else kv_size_gb
        )

        # (3) ViT activation memory for one batch item.
        #     With flash-attn: 3 projection tensors (Q, K, V) per layer, float16.
        #     We add a ×2 safety factor to cover residuals and layer-norm buffers.
        vision_cfg = self.pipe.model.config.vision_config
        vit_hidden = getattr(vision_cfg, "hidden_size", 1024)
        vit_layers = getattr(vision_cfg, "num_hidden_layers", 24)
        patch_size = getattr(vision_cfg, "patch_size", 14)
        num_patches = (336 // patch_size) ** 2 + 1  # 336 = thumbnail cap; +1 for CLS
        vit_mem_gb = (2 * 3 * num_patches * vit_hidden * 2 * vit_layers) / 1e9

        # (4) LLM activation memory for one batch item during the first generate step.
        #     Input sequence length = max_ctx + right_len (the most expensive step).
        #     With flash-attn: 3 Q/K/V projections + 2 FFN intermediate tensors, float16.
        #     We add a ×2 safety factor for residuals, layer-norm buffers, and output proj.
        text_cfg = self.pipe.model.config.text_config
        llm_hidden = getattr(text_cfg, "hidden_size", 4096)
        llm_intermediate = getattr(text_cfg, "intermediate_size", 14336)
        llm_layers = len(self._get_llm_layers())
        seq_len = max_context_tokens + max_right_tokens
        attn_per_layer = 3 * seq_len * llm_hidden * 2        # Q, K, V
        ffn_per_layer = 2 * seq_len * llm_intermediate * 2   # gate + up projections
        llm_mem_gb = (2 * (attn_per_layer + ffn_per_layer) * llm_layers) / 1e9

        total_per_item_gb = kv_mem_gb + q_mem_gb + vit_mem_gb + llm_mem_gb

        free_mem_bytes, _ = torch.cuda.mem_get_info(self.device_id)
        gpu_free_gb = free_mem_bytes / 1e9

        max_batch = int(0.9 * gpu_free_gb / total_per_item_gb)
        logger.debug(
            f"[image_join] batch size estimate: "
            f"kv={kv_mem_gb:.2f}GB q={q_mem_gb:.2f}GB "
            f"vit={vit_mem_gb:.2f}GB llm={llm_mem_gb:.2f}GB "
            f"total={total_per_item_gb:.2f}GB free={gpu_free_gb:.2f}GB → {max_batch}"
        )

        batch = 2 ** int(math.floor(math.log2(max_batch))) if max_batch >= 1 else 1
        batch = min(batch, 1024)
        logger.info(f"[image_join] Using batch size: {batch}")
        return batch

    # Convert DataFrame images to PIL Images
    @staticmethod
    def deserialize_image(b64_or_path):
        """Loads an image and downsizes it if it's too large."""
        try:
            if isinstance(b64_or_path, str) and b64_or_path.startswith("data:image"):
                # Handle base64 encoded images
                image_data = base64.b64decode(b64_or_path.split(",")[1])
                image = Image.open(BytesIO(image_data))
                image.load()  # Force load to catch truncation errors early
                image = image.convert("RGB")
            elif isinstance(b64_or_path, str):
                # Handle file paths
                image = Image.open(b64_or_path)
                image.load()  # Force load to catch truncation errors early
                image = image.convert("RGB")
            else:
                # Assume it's already a PIL Image or bytes
                if hasattr(b64_or_path, "convert"):
                    image = b64_or_path.convert("RGB")
                else:
                    image = Image.open(BytesIO(b64_or_path))
                    image.load()  # Force load to catch truncation errors early
                    image = image.convert("RGB")

            # Downsize image if too large
            if int(np.prod(image.size)) > IMAGE_MAX_PIXELS:
                ratio = np.sqrt(IMAGE_MAX_PIXELS / np.prod(image.size))
                image.thumbnail(
                    (int(image.size[0] * ratio), int(image.size[1] * ratio))
                )

            return image
        except Exception as e:
            logger.error(f"Error deserializing image {b64_or_path}: {str(e)}")
            raise

    async def _run_vanilla_inference(
        self,
        image_paths: List[str],
        question: str,
        boolean_question: bool,
    ):
        """Run vanilla inference without KV caching."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Set up multimodal processing parameters
        if boolean_question:
            context = "Answer the following question based on the image with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the image. Do not add any other comments."

        full_question = context + " " + question
        max_new_tokens = 4 if boolean_question else 64

        # Initialize processor
        processor = AutoProcessor.from_pretrained(self.model_name)
        answers = []
        log_odds = []

        # Process images one by one (could be batched for efficiency)
        for image_path in tqdm(image_paths, desc="Processing images (vanilla)"):
            try:
                # Load and deserialize image
                image = self.deserialize_image(image_path)

                # Create prompt structure
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": full_question},
                        ],
                    }
                ]

                # Apply chat template and prepare inputs
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )

                inputs = processor(images=image, text=prompt, return_tensors="pt")

                # Move inputs to device
                first_device = next(self.pipe.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.pipe.tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                logits = generated.scores[0]  # type: ignore
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
                id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")
                log_probs_0 = log_probs[:, id0]
                log_probs_1 = log_probs[:, id1]
                log_odds_1_vs_0 = log_probs_1 - log_probs_0

                # Decode the generated tokens
                decoded = self.pipe.tokenizer.batch_decode(
                    generated.sequences[:, inputs["input_ids"].shape[1] :],  # type: ignore
                    skip_special_tokens=True,
                )

                answers.extend(decoded)
                log_odds.extend(log_odds_1_vs_0.cpu().tolist())
                # Decode the generated tokens

                # Cleanup
                del inputs, generated
                torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Error processing image {image_path}: {str(e)}")
                answers.append("Not sure")

        # Create result dictionary
        result_answers = {
            img_path: answer for img_path, answer in zip(image_paths, answers)
        }
        result_log_odds = {
            img_path: log_odd for img_path, log_odd in zip(image_paths, log_odds)
        }

        return result_answers, result_log_odds

    async def _run_kv_cache_multimodal(
        self,
        column_name: str,
        image_paths: List[str],
        question: str,
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Run KV cache-based multimodal inference on the data."""

        # Use vanilla inference if gold_vanilla is enabled for CR=0 and 72b model
        if self.gold_vanilla and compression_ratio == 0.0 and "72b" in self.model_name:
            logger.info("Using gold-vanilla mode for inference (CR=0, 72b model)")
            return await self._run_vanilla_inference(
                image_paths=image_paths,
                question=question,
                boolean_question=boolean_question,
            )

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Check if caches exist and generate if not (current method: checked one by one)
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        recorded_errors = {}
        with open(
            f"{save_dir}/ERRORS.json",
            "r",
        ) as f:
            recorded_errors = json.load(f)

        cache_files = []
        image_paths_with_caches = []
        image_paths_without_caches = []
        for i, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            cache_name = self.hash_path(image_path)
            cache_filename = f"{save_dir}/cache_entry_{cache_name}.pt"

            if not os.path.exists(cache_filename):
                assert (
                    cache_filename in recorded_errors
                ), f"Cache file {cache_filename} missing but no recorded error found."
                logger.warning(
                    f"Skipping image {i} {image_path} due to previous error: {recorded_errors[cache_filename]}",
                )
                image_paths_without_caches.append(image_path)
                continue

            cache_files.append(cache_filename)
            image_paths_with_caches.append(image_path)

        if not cache_files:
            logger.warning("No valid caches or images found")
            return {}, {}

        # Set up multimodal processing parameters
        # context = "Answer the following question based on the image with 'yes' or 'no'. Do not add any other comments."
        if boolean_question:
            context = "Answer the following question based on the image with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the image. Do not add any other comments."
        question = context + " " + question
        answer_prefix = "Answer: "
        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        llm_layers = self._get_llm_layers()
        layer_devices = [
            llm_layers[i].self_attn.q_proj.weight.device for i in range(len(llm_layers))
        ]
        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            layer_devices=layer_devices,
            file_paths=cache_files,
            cache_dir=cache_dir,
        )
        max_new_tokens = 4 if boolean_question else 64

        # Initialize processor
        processor = AutoProcessor.from_pretrained(self.model_name)
        answers = []
        log_odds = []

        # Process images in batches
        for batch_start in tqdm(
            range(0, len(cache_files), batch_size),
            desc=f"Processing batches for CR {compression_ratio}",
        ):
            batch_input_ids = []
            batch_attention_masks = []
            caches = []
            context_lengths = []

            batch_images = cache_files[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_size_actual = len(batch_images)

            # Generate or load caches for each image in the batch
            for i in range(batch_size_actual):
                # Load the pre-generated cache to CPU first to avoid GPU memory accumulation
                cache = torch.load(
                    batch_images[i], map_location="cpu", weights_only=False
                )
                caches.append(cache)
                if hasattr(cache, "layers") and cache.layers:
                    context_lengths.append(cache.layers[0].keys.shape[2])
                elif hasattr(cache, "_cache") and cache._cache:
                    context_lengths.append(cache._cache[0].key_states.shape[2])
                else:
                    context_lengths.append(cache.__dict__["key_cache"][0].shape[2])

            if not caches:
                continue

            max_context_len = max(context_lengths)

            # Prepare inputs for each image in the batch
            for i, ctx_len in enumerate(context_lengths):
                padded_context_ids = torch.full(
                    (1, ctx_len),
                    self.pipe.tokenizer.pad_token_id + 1,
                    device=self.device,
                )
                pad_len = max_context_len - ctx_len
                padding_ids = torch.full(
                    (1, pad_len), self.pipe.tokenizer.pad_token_id, device=self.device
                )
                padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)

                # Create the prompt structure
                separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n" + "#"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": context + separator},
                        ],
                    }
                ]

                # Apply chat template
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                _, question_suffix = prompt.split(separator)

                question_text = question + question_suffix + answer_prefix
                question_ids = self.pipe.tokenizer.encode(
                    question_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                input_ids = torch.cat([padded_context, question_ids], dim=1)

                # Create attention masks
                context_mask = torch.ones_like(padded_context_ids)
                padding_mask = torch.zeros_like(padding_ids)
                question_mask = torch.ones_like(question_ids)
                attention_mask = torch.cat(
                    [padding_mask, context_mask, question_mask], dim=1
                )

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)

            # Batch the inputs
            batched_inputs = torch.cat(batch_input_ids, dim=0)
            batched_attention_mask = torch.cat(batch_attention_masks, dim=0)

            # Batch the caches
            batched_cache = []
            for layers in zip(*[_iter_cache_layers(c) for c in caches]):
                max_seq_len = max(k.shape[2] for k, _ in layers)
                keys_padded = []
                values_padded = []

                for k, v in layers:
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

            # Move inputs to the device of the embedding layer (first layer)
            first_device = next(self.pipe.model.parameters()).device
            batched_inputs = batched_inputs.to(first_device)
            batched_attention_mask = batched_attention_mask.to(first_device)

            # Move each cache layer to the same device as the corresponding model layer
            llm_layers = self._get_llm_layers()
            if hasattr(padded_cache, "layers"):
                for layer_idx, layer in enumerate(padded_cache.layers):
                    layer_device = llm_layers[layer_idx].self_attn.q_proj.weight.device
                    layer.keys = layer.keys.to(layer_device)
                    layer.values = layer.values.to(layer_device)
            else:
                for layer_idx, item in enumerate(padded_cache._cache):
                    layer_device = llm_layers[layer_idx].self_attn.q_proj.weight.device
                    item.key_states = item.key_states.to(layer_device)
                    item.value_states = item.value_states.to(layer_device)

            # Generate responses
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    input_ids=batched_inputs,
                    attention_mask=batched_attention_mask,
                    past_key_values=padded_cache,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            logits = generated.scores[0]  # type: ignore
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
            id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")
            log_probs_0 = log_probs[:, id0]
            log_probs_1 = log_probs[:, id1]
            log_odds_1_vs_0 = log_probs_1 - log_probs_0

            # Decode the generated tokens
            decoded = self.pipe.tokenizer.batch_decode(
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
            del batch_input_ids, batch_attention_masks, batched_cache
            torch.cuda.empty_cache()

        # Process answers and create mask
        # mask = []
        # transformed_keep_answer = "1" if keep_answer == "yes" else "0"
        # for data_id, answer in zip(data_ids, answers):
        #     predicted_answer = transformed_keep_answer in answer.lower()
        #     # logger.info(f"Row {data_id} answer: {answer} -> {predicted_answer}")
        #     mask.append((data_id, predicted_answer))

        result_answers = {
            img_path: "Not sure" for img_path in image_paths_without_caches
        }
        result_answers.update(
            {
                img_path: answer
                for img_path, answer in zip(image_paths_with_caches, answers)
            }
        )
        result_log_odds = {img_path: 0.0 for img_path in image_paths_without_caches}
        result_log_odds.update(
            {
                img_path: log_odd
                for img_path, log_odd in zip(image_paths_with_caches, log_odds)
            }
        )

        return result_answers, result_log_odds

    def compute_image_qa_join_response(
        self,
        left_column_name: str,
        pairs: List[Dict[str, str]],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ) -> Dict[str, Dict]:
        """Run join inference: image A is KV-cached, image B is passed inline."""
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        answers, log_odds = asyncio.run(
            self._run_kv_cache_multimodal_join(
                left_column_name=left_column_name,
                pairs=pairs,
                question=question,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
            )
        )
        return {"answers": answers, "log_odds": log_odds}

    async def _run_kv_cache_multimodal_join(
        self,
        left_column_name: str,
        pairs: List[Dict[str, str]],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Batched join: load B left image KV caches at once, then for each right
        image run one batch-B generate() call — mirrors the TextQA join structure.

        Outer loop: batches of B left images (KV cache on GPU, stacked to (B, heads, ctx, dim)).
        Inner loop: M unique right images, each processed as a single batch-B forward pass.
        Total generate() calls: ceil(N/B) × M  vs.  N × M  for the old per-pair approach.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"

        join_instruction = (
            "Answer the following question about the two images with '1' or '0'. "
            "Do not add any other comments."
        )
        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64

        processor = AutoProcessor.from_pretrained(self.model_name)
        first_device = next(self.pipe.model.parameters()).device
        llm_layers = self._get_llm_layers()
        layer_devices = [
            llm_layers[i].self_attn.q_proj.weight.device for i in range(len(llm_layers))
        ]
        id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
        id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")

        result_answers: Dict[str, str] = {}
        result_log_odds: Dict[str, float] = {}

        # Build ordered unique left / right lists and a fast pair-membership set.
        unique_lefts = list(dict.fromkeys(p["left"] for p in pairs))
        unique_rights = list(dict.fromkeys(p["right"] for p in pairs))
        requested_pairs = {(p["left"], p["right"]) for p in pairs}

        # Pre-build the right-side prompt template (identical for every right image).
        full_question = join_instruction + " " + question
        messages_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_question},
                ],
            }
        ]
        prompt_template = (
            processor.apply_chat_template(
                messages_template, add_generation_prompt=True, tokenize=False
            )
            + answer_prefix
        )

        # Filter left images whose cache is missing; mark those pairs as "Not sure".
        valid_lefts = []
        for left_path in unique_lefts:
            cache_filename = f"{save_dir}/cache_entry_{self.hash_path(left_path)}.pt"
            if not os.path.exists(cache_filename):
                logger.warning(
                    f"ImageQAJoin: skipping left image {left_path} — cache missing"
                )
                for right_path in unique_rights:
                    if (left_path, right_path) in requested_pairs:
                        key = f"{left_path}|{right_path}"
                        result_answers[key] = "Not sure"
                        result_log_odds[key] = 0.0
            else:
                valid_lefts.append(left_path)

        N = len(valid_lefts)
        M = len(unique_rights)

        # --- Dynamic batch size (mirrors TextQA join) ---
        # Step 1: measure right-side token count from the first right image (CPU only).
        _sample_right = self.deserialize_image(unique_rights[0])
        _sample_right.thumbnail((336, 336))
        _sample_inputs = processor(
            images=_sample_right, text=prompt_template, return_tensors="pt"
        )
        sample_right_len = _sample_inputs.input_ids.shape[1] - 1  # strip BOS
        del _sample_right, _sample_inputs

        # Step 2: measure left context length from the first valid cache (CPU load).
        _sample_cache = torch.load(
            f"{save_dir}/cache_entry_{self.hash_path(valid_lefts[0])}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if hasattr(_sample_cache, "layers") and _sample_cache.layers:
            sample_ctx_len = _sample_cache.layers[0].keys.shape[2]
        elif hasattr(_sample_cache, "_cache") and _sample_cache._cache:
            sample_ctx_len = _sample_cache._cache[0].key_states.shape[2]
        else:
            sample_ctx_len = _sample_cache.__dict__["key_cache"][0].shape[2]
        del _sample_cache

        # Step 3: compute batch size — use the explicit per-ratio override if set,
        # otherwise call the image-specific formula that accounts for ViT + LLM activations.
        cache_file_paths = [
            f"{save_dir}/cache_entry_{self.hash_path(lp)}.pt" for lp in valid_lefts
        ]
        configured = self.compression_ratio_to_batch_size.get(compression_ratio)
        if configured is not None:
            batch_size = configured
        else:
            try:
                batch_size = self._get_max_batch_size_image_join(
                    file_paths=cache_file_paths,
                    max_right_tokens=sample_right_len,
                    max_context_tokens=sample_ctx_len,
                )
            except Exception as e:
                logger.warning(
                    f"[image_join] Could not compute batch size: {e}. Falling back to 1."
                )
                batch_size = 1

        logger.info(
            f"[image_join] N={N} valid left images, M={M} right images, "
            f"batch_size={batch_size}, compression_ratio={compression_ratio}"
        )

        for batch_start in tqdm(
            range(0, N, batch_size),
            desc="ImageQA join: processing batches of left images",
        ):
            batch_lefts = valid_lefts[batch_start : batch_start + batch_size]
            B = len(batch_lefts)

            # ------------------------------------------------------------------
            # Step 1: Load B KV caches from disk, move each layer to its GPU device.
            # ------------------------------------------------------------------
            caches = []  # list of (key_layers, val_layers) — each layer already on GPU
            context_lens = []
            for left_path in batch_lefts:
                cache_filename = (
                    f"{save_dir}/cache_entry_{self.hash_path(left_path)}.pt"
                )
                cache_data = torch.load(
                    cache_filename, map_location="cpu", weights_only=False
                )
                # Normalise across DynamicCache format versions
                if hasattr(cache_data, "layers") and cache_data.layers:
                    keys = [
                        layer.keys.to(layer_devices[j])
                        for j, layer in enumerate(cache_data.layers)
                    ]
                    vals = [
                        layer.values.to(layer_devices[j])
                        for j, layer in enumerate(cache_data.layers)
                    ]
                elif hasattr(cache_data, "_cache") and cache_data._cache:
                    keys = [
                        item.key_states.to(layer_devices[j])
                        for j, item in enumerate(cache_data._cache)
                    ]
                    vals = [
                        item.value_states.to(layer_devices[j])
                        for j, item in enumerate(cache_data._cache)
                    ]
                else:
                    key_cache = cache_data.__dict__["key_cache"]
                    val_cache = cache_data.__dict__["value_cache"]
                    keys = [k.to(layer_devices[j]) for j, k in enumerate(key_cache)]
                    vals = [v.to(layer_devices[j]) for j, v in enumerate(val_cache)]
                caches.append((keys, vals))
                context_lens.append(keys[0].shape[2])
                del cache_data

            max_ctx = max(context_lens)

            # ------------------------------------------------------------------
            # Step 2: Left-pad and stack B caches → (B, heads, max_ctx, head_dim)
            # per layer.  Shorter caches get zero-padded on the left (attention
            # mask will mask those positions out during the forward pass).
            # ------------------------------------------------------------------
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
                batched_gpu_keys.append(torch.cat(keys_padded, dim=0))
                batched_gpu_values.append(torch.cat(values_padded, dim=0))

            # ------------------------------------------------------------------
            # Step 3: For each right image, run one batch-B generate() call.
            # ------------------------------------------------------------------
            for right_path in unique_rights:
                # Skip right images not paired with any left in this batch.
                if not any(
                    (left_path, right_path) in requested_pairs
                    for left_path in batch_lefts
                ):
                    continue

                try:
                    image_B = self.deserialize_image(right_path)
                    # Cap to a single tile (336×336) to avoid a transformers 5.x
                    # LLaVA-NeXT bug where the feature count disagrees with tokens.
                    image_B.thumbnail((336, 336))

                    inputs_B = processor(
                        images=image_B, text=prompt_template, return_tensors="pt"
                    )

                    # Strip leading BOS — it is already covered by the left KV cache.
                    right_ids_1 = inputs_B.input_ids[:, 1:].to(first_device)  # (1, R)
                    pixel_values_1 = inputs_B.pixel_values.to(first_device)   # (1, P, C, H, W)
                    image_sizes_1 = inputs_B.image_sizes.to(first_device)     # (1, 2)
                    right_len = right_ids_1.shape[1]

                    # Expand right-side tensors to batch size B — each left context
                    # is compared against the same right image.
                    pixel_values_B = pixel_values_1.expand(
                        B, *pixel_values_1.shape[1:]
                    ).contiguous()
                    image_sizes_B = image_sizes_1.expand(B, -1).contiguous()

                    # Build (B, max_ctx + right_len) input_ids and attention_mask.
                    # Layout per item: [left_padding | ctx_placeholder | right_tokens]
                    # Attention mask: [0…0 | 1…1 | 1…1]
                    batch_input_ids = []
                    batch_attn_mask = []
                    for ctx_len in context_lens:
                        pad_len = max_ctx - ctx_len
                        pad_ids = torch.full(
                            (1, pad_len),
                            self.pipe.tokenizer.pad_token_id,
                            device=first_device,
                        )
                        ctx_ids = torch.full(
                            (1, ctx_len),
                            self.pipe.tokenizer.pad_token_id + 1,
                            device=first_device,
                        )
                        input_ids_i = torch.cat(
                            [pad_ids, ctx_ids, right_ids_1.expand(1, -1)], dim=1
                        )
                        attn_mask_i = torch.cat(
                            [
                                torch.zeros_like(pad_ids),
                                torch.ones_like(ctx_ids),
                                torch.ones((1, right_len), device=first_device),
                            ],
                            dim=1,
                        )
                        batch_input_ids.append(input_ids_i)
                        batch_attn_mask.append(attn_mask_i)

                    batched_inputs = torch.cat(batch_input_ids, dim=0)  # (B, max_ctx+R)
                    batched_attn = torch.cat(batch_attn_mask, dim=0)    # (B, max_ctx+R)

                    # Fresh DynamicCache for this forward pass — generate() mutates
                    # it in-place, so we must not reuse across right images.
                    batch_cache = DynamicCache()
                    for layer_idx, (k, v) in enumerate(
                        zip(batched_gpu_keys, batched_gpu_values)
                    ):
                        batch_cache.update(k.clone(), v.clone(), layer_idx)

                    with torch.no_grad():
                        generated = self.pipe.model.generate(
                            input_ids=batched_inputs,
                            attention_mask=batched_attn,
                            past_key_values=batch_cache,
                            pixel_values=pixel_values_B,
                            image_sizes=image_sizes_B,
                            pad_token_id=self.pipe.tokenizer.eos_token_id,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                    logits = generated.scores[0]  # (B, vocab_size)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    log_odds_vals = (log_probs[:, id1] - log_probs[:, id0]).tolist()

                    decoded_all = self.pipe.tokenizer.batch_decode(
                        generated.sequences[:, batched_inputs.shape[1] :],
                        skip_special_tokens=True,
                    )

                    for b, left_path in enumerate(batch_lefts):
                        if (left_path, right_path) in requested_pairs:
                            key = f"{left_path}|{right_path}"
                            result_answers[key] = (
                                decoded_all[b] if decoded_all[b] else "Not sure"
                            )
                            result_log_odds[key] = log_odds_vals[b]

                    del inputs_B, right_ids_1, pixel_values_1, image_sizes_1
                    del pixel_values_B, image_sizes_B
                    del batch_input_ids, batch_attn_mask, batched_inputs, batched_attn
                    del batch_cache, generated, logits, log_probs
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(
                        f"ImageQAJoin: error processing right image {right_path} "
                        f"against batch {batch_start}–{batch_start+B-1}: {e}"
                    )
                    for left_path in batch_lefts:
                        if (left_path, right_path) in requested_pairs:
                            key = f"{left_path}|{right_path}"
                            result_answers[key] = "Not sure"
                            result_log_odds[key] = 0.0

            # Clean up batch GPU tensors
            del caches, batched_gpu_keys, batched_gpu_values
            torch.cuda.empty_cache()

        return result_answers, result_log_odds

    def compute_text_only_qa_response(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
    ) -> Dict:
        """Synchronous wrapper for _run_text_only_qa."""
        log_odds = asyncio.run(
            self._run_text_only_qa(
                questions=questions,
                contexts=contexts,
                boolean_question=boolean_question,
            )
        )
        return {"log_odds": log_odds}

    async def _run_text_only_qa(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
    ) -> List[float]:
        """Text-only LLaVA inference — no images, no KV caches.

        For each (context, question) pair, builds a text prompt and runs generate().
        Returns log_odds as a list ordered identically to `questions`.
        """
        assert len(questions) == len(contexts)
        if boolean_question:
            instruction = (
                "Answer the following question with '1' or '0'. "
                "Do not add any other comments."
            )
        else:
            instruction = "Answer the following question. Do not add any other comments."
        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64

        processor = AutoProcessor.from_pretrained(self.model_name)
        first_device = next(self.pipe.model.parameters()).device
        id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
        id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")

        log_odds_out = []
        for context, question in zip(contexts, questions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{context}\n{instruction}\n{question}"},
                    ],
                }
            ]
            prompt = (
                processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                + answer_prefix
            )

            inputs = processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.pipe.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            logits = generated.scores[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            lo = (log_probs[0, id1] - log_probs[0, id0]).item()
            log_odds_out.append(lo)

            del inputs, generated, logits, log_probs
            torch.cuda.empty_cache()

        return log_odds_out

    async def _generate_cache_for_image(self, image, cache_filename, compression_ratio):
        """Generate and save multimodal KV cache for a single image."""
        assert self.pipe is not None, "Model pipeline is not initialized."

        try:
            # Standard context and answer format
            context = " "
            answer_prefix = "Answer: "

            # Generate cache
            inputs = self.pipe.preprocess(
                context=context,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
                image=image,
            )

            cache = DynamicCache()
            press = self.presses[compression_ratio]
            with torch.inference_mode():
                with (
                    press(self.pipe.model)
                    if press is not None
                    else contextlib.nullcontext()
                ):
                    _ = self.pipe._forward(inputs, press=press, cache=cache)

            # Save cache to disk — move to CPU in-place
            if hasattr(cache, "layers"):
                for layer in cache.layers:
                    layer.keys = layer.keys.detach().cpu()
                    layer.values = layer.values.detach().cpu()
            else:
                for item in cache._cache:
                    item.key_states = item.key_states.detach().cpu()
                    item.value_states = item.value_states.detach().cpu()

            logger.info(f"Saving cache to: {cache_filename}")

            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            torch.save(cache, cache_filename)

            if not os.path.exists(cache_filename):
                logger.error(f"Cache file was not actually created: {cache_filename}")

            # Cleanup
            del cache
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating cache for item {cache_filename}: {str(e)}")
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
            "image_paths": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
            ],
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        img_paths = data["image_paths"]
        cache_dir = data["cache_dir"]
        compression_ratio = data["compression_ratio"]
        assert model_wrapper is not None
        asyncio.run(
            model_wrapper.prepare_caches(
                column_name=column_name,
                image_paths=img_paths,
                cache_dir=cache_dir,
                compression_ratio=compression_ratio,
            )
        )
        return {"status": "cache_ready"}, 200


class ImageQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "image_paths": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
            ],
            "question": "How many cats are there?",
            "compression_ratio": 0.5,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir",

        }
        """
        data = request.get_json(force=True)
        img_paths = data["image_paths"]
        column_name = data["column_name"]
        question = data["question"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        responses = model_wrapper.compute_image_qa_response(
            column_name=column_name,
            image_paths=img_paths,
            question=question,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


class ImageQADirect(Resource):
    def post(self):
        """Evaluate images without preparing, reading, or writing KV-cache files."""
        data = request.get_json(force=True)
        assert model_wrapper is not None
        return model_wrapper.compute_image_qa_direct_response(
            image_paths=data["image_paths"],
            question=data["question"],
            boolean_question=data["boolean"],
        ), 200


class ImageQAJoin(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "left_column_name": "artworks.image",
            "pairs": [
                {"left": "/path/to/a.jpg", "right": "/path/to/b.jpg"},
                ...
            ],
            "question": "Do both images show a painting by the same artist?",
            "compression_ratio": 0.9,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir"
        }
        Returns:
        {
            "answers": {"left|right": "1", ...},
            "log_odds": {"left|right": 0.42, ...}
        }
        """
        data = request.get_json(force=True)
        left_column_name = data["left_column_name"]
        pairs = data["pairs"]
        question = data["question"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        responses = model_wrapper.compute_image_qa_join_response(
            left_column_name=left_column_name,
            pairs=pairs,
            question=question,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


class TextQADirect(Resource):
    def post(self):
        """
        Expects JSON:
        {
            "questions": ["Is it the same style as 'Impressionism'?", ...],
            "contexts": ["Cubism", ...],
            "boolean": true
        }
        Returns: {"log_odds": [0.42, ...]}
        """
        data = request.get_json(force=True)
        assert model_wrapper is not None
        responses = model_wrapper.compute_text_only_qa_response(
            questions=data["questions"],
            contexts=data["contexts"],
            boolean_question=data["boolean"],
        )
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(ImageQA, "/image_qa")
api.add_resource(ImageQADirect, "/image_qa_direct")
api.add_resource(ImageQAJoin, "/image_qa_join")
api.add_resource(PrepareCaches, "/prepare_caches")
api.add_resource(TextQADirect, "/text_qa_direct")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        choices=sorted(PORT_KV_VISION.keys()),
        help="Name of the model to use",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    parser.add_argument(
        "--gold-vanilla",
        action="store_true",
        help="Use vanilla inference (no KV caching) for CR=0 with 72b models",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP port; defaults to the model's standard ReasonDB port.",
    )
    args = parser.parse_args()
    device_id = args.device_id
    gold_vanilla = args.gold_vanilla
    model_wrapper = KvImageQaModelWrapper(
        args.model_name,
        device_id,
        compression_ratios=(0.0, 0.5, 0.8, 0.9, 0.99),
        gold_vanilla=gold_vanilla,
    )
    app.run(
        host="127.0.0.1",
        port=args.port or PORT_KV_VISION.get(args.model_name),
        debug=False,
        threaded=False,
    )
