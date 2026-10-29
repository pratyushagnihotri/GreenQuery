from typing import Optional, Sequence
import yaml
import torch
import logging
import math

from reasondb.memory_footprint.memory_report import (
    compute_memory_footprints,
    get_yaml_path,
)

logger = logging.getLogger(__name__)


class KVCachingBackendBase:
    def _get_max_batch_size(
        self,
        column_name: str,
        batch_size: Optional[int],
        compression_ratio: float,
        cache_dir: str,
        file_paths: Sequence[str],
        layer_devices: Sequence["torch.device"],
        model_name: Optional[str] = None,
        max_question_tokens: int = 0,
        max_context_tokens: int = 0,
    ) -> int:
        """Estimate the largest safe batch size given available GPU memory.

        KV caches are distributed across layer_devices (one shard per GPU), so the
        per-item cost on each GPU is kv_size_total / num_gpus.  The bottleneck GPU
        (minimum free memory) determines the maximum batch size.

        When max_question_tokens and max_context_tokens are provided the method also
        accounts for the activation memory of the query tokens processed on top of
        the KV cache during the forward pass.
        """
        yaml_path = get_yaml_path(cache_dir)
        if batch_size is not None:
            return batch_size

        # Use model name from self if not provided
        if model_name is None and hasattr(self, "model_name"):
            model_name = self.model_name

        # Hardcoded safe batch sizes for the 72B model (GPU nearly full with weights).
        _72B_BATCH_SIZES = {
            0.99: 64,
            0.9:  32,
            0.5:  16,
            0.0:  8,
        }
        if model_name == "llava-hf/llava-next-72b-hf" and compression_ratio in _72B_BATCH_SIZES:
            bs = _72B_BATCH_SIZES[compression_ratio]
            logging.info(f"Using hardcoded batch size for 72B model CR={compression_ratio}: {bs}")
            return bs

        # Retrieve max file size
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        dataset = data.get(column_name)
        if dataset is None:
            logger.warning(
                f"Column '{column_name}' not found in YAML file. Computing memory footprints on the fly..."
            )
            dataset = compute_memory_footprints(
                cache_dir,
                column_name,
                [f.split("/")[-1] for f in file_paths],
                store=False,
                model_name=model_name,
            )[column_name]

        # If model_name is provided, look for data for that specific model
        compression_data = dataset[model_name]
        if compression_ratio not in compression_data:
            raise KeyError(
                f"Key '{compression_ratio}' not found in dataset '{column_name}'. Available keys: {compression_data.keys()}"
            )

        kv_size_gb = compression_data[compression_ratio]
        num_gpus = len(layer_devices)

        # KV caches are sharded evenly across GPUs; each GPU holds 1/num_gpus of the
        # total.  3× loading overhead applies to the per-GPU share.
        kv_mem_per_gpu = 12 * kv_size_gb / num_gpus

        # Query-token activations are likewise distributed across all GPUs.
        if max_question_tokens > 0 and max_context_tokens > 0:
            q_mem_per_gpu = (
                kv_size_gb * (max_question_tokens / max_context_tokens)
            ) / num_gpus
        else:
            q_mem_per_gpu = 0.0

        # Forward-pass MLP activations are not captured by KV size.  For very
        # compressed caches (e.g. CR=0.99) kv_size_gb is near zero, which would
        # otherwise produce an unrealistically large batch estimate.  Clamp to a
        # minimum of 0.5 GB per item per GPU to stay safe with large models.
        total_mem_per_item_per_gpu = max(kv_mem_per_gpu + q_mem_per_gpu, 0.5 / num_gpus)

        # Use the GPU with the least free memory as the bottleneck.
        min_free_bytes = min(torch.cuda.mem_get_info(d)[0] for d in layer_devices)
        min_free_gb = min_free_bytes / 1e9
        max_batch = int(0.9 * min_free_gb // total_mem_per_item_per_gpu)
        logger.debug(
            f"Batch size estimate ({num_gpus} GPU(s)): "
            f"kv/gpu={kv_mem_per_gpu:.2f}GB q/gpu={q_mem_per_gpu:.2f}GB "
            f"total/gpu={total_mem_per_item_per_gpu:.2f}GB min_free={min_free_gb:.2f}GB → {max_batch}"
        )

        # Largest power of 2 <= max_batch
        batch = 2 ** int(math.floor(math.log2(max_batch))) if max_batch >= 1 else 1
        if batch > 1024:
            batch = 1024  # Cap batch size at 1024
        logging.info(f"Using batch size: {batch}")

        return int(max(1, batch/2))

    def _get_max_batch_size_join(
        self,
        column_name: str,
        batch_size: Optional[int],
        compression_ratio: float,
        cache_dir: str,
        file_paths: Sequence[str],
        layer_devices: Sequence["torch.device"],
        max_question_tokens: int,
        max_context_tokens: int,
        model_name: Optional[str] = None,
    ) -> int:
        """Like _get_max_batch_size, but for join-mode forward passes.

        In join mode each forward pass receives (B, max_ctx + q_len) tokens, so
        the question-token activations add memory proportional to q_len / ctx_len
        on top of the KV-cache loading cost.  Both costs are split across GPUs.
        """
        yaml_path = get_yaml_path(cache_dir)
        if batch_size is not None:
            return batch_size

        if model_name is None and hasattr(self, "model_name"):
            model_name = self.model_name

        _72B_BATCH_SIZES = {
            0.99: 64,
            0.9:  32,
            0.5:  16,
            0.0:  8,
        }
        if model_name == "llava-hf/llava-next-72b-hf" and compression_ratio in _72B_BATCH_SIZES:
            bs = _72B_BATCH_SIZES[compression_ratio]
            logging.info(f"[join] Using hardcoded batch size for 72B model CR={compression_ratio}: {bs}")
            return bs

        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        dataset = data.get(column_name)
        if dataset is None:
            logger.warning(
                f"Column '{column_name}' not found in YAML file. Computing memory footprints on the fly..."
            )
            dataset = compute_memory_footprints(
                cache_dir,
                column_name,
                [f.split("/")[-1] for f in file_paths],
                store=False,
                model_name=model_name,
            )[column_name]

        compression_data = dataset[model_name]
        if compression_ratio not in compression_data:
            raise KeyError(
                f"Key '{compression_ratio}' not found in dataset '{column_name}'. "
                f"Available keys: {compression_data.keys()}"
            )

        kv_size_gb = compression_data[compression_ratio]
        num_gpus = len(layer_devices)

        kv_mem_per_gpu = 10 * kv_size_gb / num_gpus

        if max_context_tokens > 0:
            q_mem_per_gpu = (
                kv_size_gb * (max_question_tokens / max_context_tokens)
            ) / num_gpus
        else:
            q_mem_per_gpu = (
                kv_size_gb / num_gpus
            )  # fallback: assume question ≈ context length

        total_mem_per_item_per_gpu = max(kv_mem_per_gpu + q_mem_per_gpu, 0.5 / num_gpus)

        min_free_bytes = min(torch.cuda.mem_get_info(d)[0] for d in layer_devices)
        min_free_gb = min_free_bytes / 1e9
        max_batch = int(0.9 * min_free_gb // total_mem_per_item_per_gpu)
        logger.debug(
            f"[join] Batch size estimate ({num_gpus} GPU(s)): "
            f"kv/gpu={kv_mem_per_gpu:.2f}GB q/gpu={q_mem_per_gpu:.2f}GB "
            f"total/gpu={total_mem_per_item_per_gpu:.2f}GB min_free={min_free_gb:.2f}GB → {max_batch}"
        )

        batch = 2 ** int(math.floor(math.log2(max_batch))) if max_batch >= 1 else 1
        if batch > 1024:
            batch = 1024
        logging.info(f"[join] Using batch size: {batch}")

        return int(max(1, batch/2)) 
