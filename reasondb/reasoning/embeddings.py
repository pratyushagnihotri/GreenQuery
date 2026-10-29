import os
from abc import abstractmethod
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional
from openai import AsyncOpenAI
import numpy as np

from reasondb.utils.cache import CACHE_DIR
from reasondb.utils.logging import FileLogger


EMBEDDING_CACHE_DIR = CACHE_DIR / Path("embedding_cache")


@dataclass
class EmbeddingCharacteristics:
    rpm: int
    tpm: int
    out_dim: int
    in_len: int
    cost: float  # per million tokens


class EmbeddingModel:
    def __init__(self, model_id, characteristics: EmbeddingCharacteristics):
        self.characteristics = characteristics
        self.model_id = model_id

    async def prepare(self):
        pass

    async def close(self):
        pass


class TextEmbeddingModel(EmbeddingModel):
    def _cache_key(self, prompt: str) -> str:
        # Existing caches were written by text-embedding-3-small keyed on the
        # prompt alone; other models include their id so vectors never mix.
        if self.model_id == "text-embedding-3-small":
            return hashlib.sha256(str(prompt).encode()).hexdigest()
        return hashlib.sha256(f"{self.model_id}\n{prompt}".encode()).hexdigest()

    async def embed(self, text: str, logger: FileLogger) -> np.ndarray:
        logger.debug(__name__, f"Invoking Embedding Model with prompt: {text}")
        response = None
        response: Optional[np.ndarray] = self.get_cached(text)
        if response is None:
            logger.debug(__name__, "Invoking EmbeddingModel")
            response = await self._embed(text)
            self.cache(text, response)
        else:
            logger.debug(__name__, "Using cached embedding")
        logger.debug(__name__, f"LLM response: {response}")
        return response

    def cache(self, prompt: str, response: np.ndarray):
        hash = self._cache_key(prompt)
        EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)
        path = EMBEDDING_CACHE_DIR / hash
        with open(path, "w") as f:
            json.dump({"prompt": str(prompt), "response": response.tolist()}, f)

    def get_cached(self, prompt: str) -> Optional[np.ndarray]:
        hash = self._cache_key(prompt)
        path = EMBEDDING_CACHE_DIR / hash
        if not path.exists():
            return None
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return None
        if data["prompt"] == str(prompt):
            return np.array(data["response"])

    @abstractmethod
    async def _embed(self, text: str) -> np.ndarray:
        raise NotImplementedError


class OpenAIEmbeddingModel(TextEmbeddingModel):
    def __init__(self, model_id: str, characteristics):
        super().__init__(model_id, characteristics)
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client: Optional[AsyncOpenAI] = None

    async def prepare(self):
        await super().prepare()
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def close(self):
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def _embed(self, text: str) -> np.ndarray:
        assert self.client is not None, "OpenAI client not initialized"
        response = await self.client.embeddings.create(
            input=text[: self.characteristics.in_len], model=self.model_id
        )
        assert response.data[0].embedding is not None
        return np.array(response.data[0].embedding)


class TextEmbedding3Small(OpenAIEmbeddingModel):
    def __init__(self):
        super().__init__(
            model_id="text-embedding-3-small",
            characteristics=EmbeddingCharacteristics(
                rpm=10_000, tpm=10_000_000, out_dim=1536, in_len=8189, cost=0.02
            ),
        )


class LocalTextEmbedding(TextEmbeddingModel):
    """Hugging Face sentence embedding run in-process (no API key needed)."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", out_dim: int = 384):
        super().__init__(
            model_id=model_id,
            characteristics=EmbeddingCharacteristics(
                rpm=1_000_000, tpm=1_000_000_000, out_dim=out_dim, in_len=2048, cost=0.0
            ),
        )
        self._tokenizer = None
        self._model = None

    async def prepare(self):
        await super().prepare()
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).eval()

    def _encode(self, text: str) -> np.ndarray:
        import torch

        assert self._tokenizer is not None and self._model is not None, "Prepare embedding model first"
        inputs = self._tokenizer(
            text[: self.characteristics.in_len], truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            # BGE models use the normalized [CLS] vector as the sentence embedding.
            vector = self._model(**inputs).last_hidden_state[0, 0]
        vector = torch.nn.functional.normalize(vector, dim=0)
        return vector.cpu().numpy()

    async def _embed(self, text: str) -> np.ndarray:
        import asyncio

        return await asyncio.to_thread(self._encode, text)


def default_embedding_model() -> TextEmbeddingModel:
    """GREENQUERY_EMBEDDING_MODEL = openai | local | auto (OpenAI when a key is set)."""
    choice = os.environ.get("GREENQUERY_EMBEDDING_MODEL", "auto")
    if choice == "auto":
        choice = "openai" if os.environ.get("OPENAI_API_KEY") else "local"
    if choice == "openai":
        return TextEmbedding3Small()
    if choice == "local":
        return LocalTextEmbedding()
    raise ValueError(f"Unknown GREENQUERY_EMBEDDING_MODEL: {choice}")
