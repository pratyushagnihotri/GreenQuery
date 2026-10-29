import logging
import argparse
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from reasondb.backends.text_embeddings import PORT_TEXT_SIM

# SOTA model for short text embeddings (MTEB Leaderboard favorite)
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32
DATALOADER_NUM_WORKERS = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


class TextEmbeddingModelWrapper:
    def __init__(self, model_name, device_id):
        self.model_name = model_name
        self.device_id = device_id
        self.device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.init_model()

    def init_model(self):
        logger.info(f"Loading Text Embedding model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension dynamically
        self.embed_dim = self.model.config.hidden_size
        logger.info(f"Model loaded. Dimension: {self.embed_dim}")

    def compute_batch_embeddings(self, texts: List[str]):
        # Tokenize
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. BGE uses the [CLS] token (index 0)
            embeddings = model_output[0][:, 0]
            # Normalize embeddings (standard practice for BGE/Cosine Similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_single_embedding(self, text: str):
        """Helper for single query strings."""
        encoded_input = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embedding = model_output[0][:, 0]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding[0].cpu().numpy()


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "embed_dim": model_wrapper.embed_dim,
            "device": str(model_wrapper.device),
        }, 200


class TextSimilarity(Resource):
    def post(self):
        """
        Expects JSON:
        {
            "texts1": ["Text", ...],
            "texts2": ["Text", ...]
        }
        """
        data = request.get_json(force=True)
        texts1 = data.get("texts1", [])
        texts2 = data.get("texts2", [])

        texts_unique = list(set(texts1 + texts2))

        assert model_wrapper is not None
        embeddings = model_wrapper.compute_batch_embeddings(texts_unique)
        device = embeddings.device

        texts_map = {t: i for i, t in enumerate(texts_unique)}
        idx1 = torch.tensor(
            [texts_map[t] for t in texts1], dtype=torch.int, device=device
        )
        idx2 = torch.tensor(
            [texts_map[t] for t in texts2], dtype=torch.int, device=device
        )
        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]

        similarities = (emb1 * emb2).sum(1).cpu().tolist()
        return {"similarities": similarities}, 200


api.add_resource(Status, "/status")
api.add_resource(TextSimilarity, "/text_sim")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    model_wrapper = TextEmbeddingModelWrapper(MODEL_NAME, args.device_id)
    app.run(host="127.0.0.1", port=PORT_TEXT_SIM, debug=False)
