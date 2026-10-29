import requests
import os
from datetime import datetime
from typing import Tuple, List

from reasondb.backends.backend import Backend
from reasondb.utils.logging import FileLogger


BATCH_SIZE = 100
PORT_TEXT_SIM = int(os.getenv("GREENQUERY_TEXT_EMBED_PORT", "5324"))


class TextSimilarityBackend(Backend):
    def __init__(self, model_name: str):
        self.check_model_name = model_name
        self.embed_dim = 0

    def setup(self, logger: FileLogger):
        result = requests.get(f"http://localhost:{PORT_TEXT_SIM}/status")
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self.check_model_name

        self.embed_dim = json_response["embed_dim"]
        logger.info(
            "__name__",
            f"Text similarity model {self.check_model_name} is ready (Dim: {self.embed_dim})",
        )

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    async def run(
        self,
        texts1: List[str],
        texts2: List[str],
    ) -> Tuple[List[float], float, float]:
        """Generates an embedding for a search query."""
        start_time = datetime.now()
        result = requests.post(
            f"http://localhost:{PORT_TEXT_SIM}/text_sim",
            json={"texts1": texts1, "texts2": texts2},
        )
        assert result.status_code == 200
        result_json = result.json()

        similarities = result_json["similarities"]
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        return similarities, runtime, 0.0

    def get_operation_identifier(self) -> str:
        return f"TextSimilarityBackend-{self.check_model_name}"

    def setup_index(self, table, column):
        pass
