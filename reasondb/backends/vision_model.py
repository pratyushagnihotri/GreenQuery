import asyncio
from copy import copy
import json
import requests
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import os

from reasondb.database.indentifier import ConcreteColumnIdentifier
from reasondb.reasoning.llm import LargeLanguageModel, Message, Prompt
from reasondb.utils.cache import CACHE_DIR
from reasondb.utils.logging import FileLogger


VISION_MODEL_CACHE_DIR = CACHE_DIR / Path("vision_model_cache")
PORT_VISION = 5006
PORT_KV_VISION = {
    "llava-hf/llava-next-72b-hf": 5008,
    "llava-hf/llama3-llava-next-8b-hf": 5009,
}


def vision_qa_batch_size() -> int:
    """Return the bounded local vision inference batch size."""
    value = os.environ.get("GREENQUERY_VISION_QA_BATCH_SIZE", "8")
    try:
        batch_size = int(value)
    except ValueError as error:
        raise ValueError("GREENQUERY_VISION_QA_BATCH_SIZE must be a positive integer") from error
    if batch_size < 1:
        raise ValueError("GREENQUERY_VISION_QA_BATCH_SIZE must be a positive integer")
    return batch_size


def vision_server_port(model_id: str) -> int:
    """Use a per-job override when a Medical suite runs beside another job."""
    configured = os.environ.get("GREENQUERY_LOCAL_VISION_PORT")
    return int(configured) if configured else PORT_KV_VISION[model_id]


@dataclass
class VisionModelCharacteristics:
    batch_size: int
    rpm: int
    tpm: int
    out_len: int
    in_len: int
    in_cost: float  # per million tokens
    out_cost: float  # per million tokens


CHARACTERISTICS_DICT = {
    "Salesforce/blip2-opt-2.7b": VisionModelCharacteristics(
        batch_size=1024,
        rpm=0,
        tpm=0,
        out_len=0,
        in_len=0,
        in_cost=0,
        out_cost=0,
    ),
    "llava-hf/llama3-llava-next-8b-hf": VisionModelCharacteristics(
        batch_size=1024,
        rpm=0,
        tpm=0,
        out_len=0,
        in_len=0,
        in_cost=0,
        out_cost=0,
    ),
    "llava-hf/llava-next-72b-hf": VisionModelCharacteristics(
        batch_size=1024,
        rpm=0,
        tpm=0,
        out_len=0,
        in_len=0,
        in_cost=0,
        out_cost=0,
    ),
}


@dataclass
class VisionModelOutputItem:
    image_path: Path
    response: str
    log_odds: float
    runtime: float
    cost: float


class VisionModel(ABC):
    def __init__(self, model_id, characteristics: VisionModelCharacteristics):
        self.characteristics = characteristics
        self._model_id = model_id

    @property
    @abstractmethod
    def cache_enabled(self) -> bool:
        pass

    @abstractmethod
    def setup(self, logger: FileLogger):
        pass

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    @abstractmethod
    def returns_log_odds(self) -> bool:
        pass

    async def invoke(
        self,
        column: ConcreteColumnIdentifier,
        image_paths: List[Path],
        question: str,
        boolean_question: bool,
        cache_dir: Path,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        logger.debug(__name__, f"Invoking VisionModel with question: {question}")
        non_cached = []
        result: Dict[Path, VisionModelOutputItem] = {}

        for image_path in image_paths:
            item = None
            if self.cache_enabled:
                item = self.get_cached(question, image_path)
            if item is None:
                non_cached.append(image_path)
            else:
                result[image_path] = item
                logger.debug(__name__, "Using cached response")

        for i in range(0, len(non_cached), self.characteristics.batch_size):
            batch = non_cached[i : i + self.characteristics.batch_size]
            logger.debug(__name__, "Invoking VisionModel")
            invokation_result = await self._invoke(
                column=column,
                question=question,
                image_paths=batch,
                cache_dir=cache_dir,
                boolean_question=boolean_question,
                logger=logger,
            )
            for item in invokation_result:
                if self.cache_enabled:
                    self.cache(question, item)
                result[item.image_path] = item
        return [result[image_path] for image_path in image_paths]

    def cache(
        self,
        question: str,
        item: VisionModelOutputItem,
    ):
        hash = hashlib.sha256(f"{question}-{item.image_path}".encode()).hexdigest()
        path = VISION_MODEL_CACHE_DIR / self.model_id / hash
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "question": str(question),
                    "image_path": str(item.image_path),
                    "response": str(item.response),
                    "log_odds": item.log_odds,
                    "runtime": item.runtime,
                    "cost": item.cost,
                },
                f,
            )

    @abstractmethod
    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        pass

    @abstractmethod
    async def wind_down(self):
        pass

    def get_cached(
        self, question: str, image_path: Path
    ) -> Optional[VisionModelOutputItem]:
        hash = hashlib.sha256(f"{question}-{image_path}".encode()).hexdigest()
        path = VISION_MODEL_CACHE_DIR / self.model_id / hash
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        if data["question"] == str(question) and data["image_path"] == str(image_path):
            return VisionModelOutputItem(
                image_path=image_path,
                response=data["response"],
                log_odds=data["log_odds"],
                runtime=data["runtime"],
                cost=data["cost"],
            )

    @abstractmethod
    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        raise NotImplementedError


class LocalVisionModel(VisionModel):
    def __init__(self, model_id):
        characteristics = CHARACTERISTICS_DICT[model_id]
        super().__init__(model_id, characteristics)

    @property
    def cache_enabled(self) -> bool:
        return True

    @property
    def returns_log_odds(self) -> bool:
        return False

    def setup(
        self,
        logger: FileLogger,
    ):
        result = requests.get(f"http://localhost:{PORT_VISION}/status")
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self.model_id
        logger.info(
            __name__,
            f"Image QA model {self.model_id} is ready",
        )

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_VISION}/image_qa",
            json={"image_paths": [str(p) for p in image_paths], "question": question},
        )
        assert response.status_code == 200
        json_response = response.json()
        time_end = time.time()
        runtime = time_end - time_start
        cost = 0.0  
        result: List[VisionModelOutputItem] = []
        for image_path in image_paths:
            result_text = json_response.get(str(image_path), "Not sure")
            result.append(
                VisionModelOutputItem(
                    image_path=image_path,
                    response=result_text,
                    log_odds=0.0,
                    runtime=runtime / len(image_paths),
                    cost=cost,
                )
            )
        return result

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        pass

    async def wind_down(self):
        pass


class KvVisionModel(VisionModel):
    def __init__(self, model_id, compression_ratio: float, cache_enabled: bool = True):
        characteristics = CHARACTERISTICS_DICT[model_id]
        self.compression_ratio = compression_ratio
        self._cache_enabled = cache_enabled
        characteristics.batch_size = vision_qa_batch_size()
        super().__init__(model_id, characteristics)

    @property
    def cache_enabled(self) -> bool:
        return self._cache_enabled

    @property
    def returns_log_odds(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return f"{self._model_id}-cr{str(self.compression_ratio)}"

    def setup(
        self,
        logger: FileLogger,
    ):
        result = requests.get(
            f"http://localhost:{vision_server_port(self._model_id)}/status"
        )
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self._model_id
        assert self.compression_ratio in json_response["compression_ratios"]
        logger.info(
            __name__,
            f"KV Vision model {self.model_id} with compression ratio {self.compression_ratio} is ready",
        )

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        if not self.cache_enabled:
            return
        response = requests.post(
            f"http://localhost:{vision_server_port(self._model_id)}/prepare_caches",
            json={
                "column_name": column.name,
                "image_paths": [str(p) for p in image_paths],
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-image-qa-cache",
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "cache_ready"

    async def wind_down(self):
        pass

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        time_start = time.time()
        endpoint = "image_qa" if self.cache_enabled else "image_qa_direct"
        response = requests.post(
            f"http://localhost:{vision_server_port(self._model_id)}/{endpoint}",
            json={
                "column_name": column.name,
                "image_paths": [str(p) for p in image_paths],
                "question": question,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-image-qa-cache",
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        answers = json_response.get("answers", {})
        log_odds = json_response.get("log_odds", {})
        time_end = time.time()
        runtime = time_end - time_start
        telemetry = json_response.get("telemetry")
        if telemetry:
            telemetry.update(
                {
                    "event_kind": "vision",
                    "execution_mode": "direct_no_cache" if not self.cache_enabled else "kv_cache",
                    "inference_batch_size": self.characteristics.batch_size,
                    "request_count": 1,
                    "processed_rows": len(image_paths),
                    "answer_cache_hits": 0,
                    "kv_cache_hits": 0 if not self.cache_enabled else telemetry.get("kv_cache_hits", 0),
                    "inference_started_at": time_start,
                    "inference_finished_at": time_end,
                    "inference_seconds": runtime,
                }
            )
            try:
                from evaluation.measurement.local_inference import record_local_inference

                record_local_inference(telemetry)
            except Exception:
                pass
        result = []
        cost = 0.0
        for image_path in image_paths:
            result_text = answers.get(str(image_path), "Not sure")
            lo = log_odds.get(str(image_path), 0.0)
            result.append(
                VisionModelOutputItem(
                    image_path=image_path,
                    response=result_text,
                    log_odds=lo,
                    runtime=runtime / len(image_paths),
                    cost=cost,
                )
            )
        return result

    async def invoke_text_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        """Call /text_qa_direct: text-only LLaVA inference (no images, no KV caches)."""
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{vision_server_port(self._model_id)}/text_qa_direct",
            json={
                "questions": questions,
                "contexts": contexts,
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        log_odds_list = response.json()["log_odds"]
        runtime = time.time() - time_start
        return [
            VisionModelOutputItem(
                image_path=Path(""),
                response="",
                log_odds=lo,
                runtime=runtime / max(len(questions), 1),
                cost=0.0,
            )
            for lo in log_odds_list
        ]

    async def invoke_join(
        self,
        left_column: ConcreteColumnIdentifier,
        pairs: List[Tuple[Path, Path]],
        question: str,
        boolean_question: bool,
        cache_dir: Path,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        """Call /image_qa_join: left image KV-cached, right image inline."""
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{vision_server_port(self._model_id)}/image_qa_join",
            json={
                "left_column_name": left_column.name,
                "pairs": [{"left": str(l), "right": str(r)} for l, r in pairs],
                "question": question,
                "compression_ratio": self.compression_ratio,
                "boolean": boolean_question,
                "cache_dir": str(cache_dir) + "/kv-image-qa-cache",
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        answers = json_response.get("answers", {})
        log_odds_map = json_response.get("log_odds", {})
        time_end = time.time()
        runtime = time_end - time_start
        cost = 0.0
        result = []
        for left_path, right_path in pairs:
            key = f"{left_path}|{right_path}"
            result_text = answers.get(key, "Not sure")
            lo = log_odds_map.get(key, 0.0)
            result.append(
                VisionModelOutputItem(
                    image_path=right_path,
                    response=result_text,
                    log_odds=lo,
                    runtime=runtime / max(len(pairs), 1),
                    cost=cost,
                )
            )
        return result


class LlmVisionModel(VisionModel):
    def __init__(self, llm: LargeLanguageModel):
        self._cache_enabled = llm.cache_enabled
        self.llm = copy(llm)
        self.llm.cache_enabled = False
        self._prepared = False
        self.characteristics = VisionModelCharacteristics(
            batch_size=50,
            rpm=llm.characteristics.rpm,
            tpm=llm.characteristics.tpm,
            out_len=llm.characteristics.out_len,
            in_len=llm.characteristics.in_len,
            in_cost=llm.characteristics.in_cost,  
            out_cost=llm.characteristics.out_cost,
        )

    @property
    def returns_log_odds(self) -> bool:
        return False

    @property
    def cache_enabled(self) -> bool:
        return self._cache_enabled

    @property
    def model_id(self) -> str:
        return self.llm.model_id

    def setup(
        self,
        logger: FileLogger,
    ):
        pass

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        await self.llm.prepare()
        self._prepared = True

    async def wind_down(self):
        await self.llm.close()
        self._prepared = False

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        if not self._prepared:
            await self.llm.prepare()
            self._prepared = True
        extended_question = question
        if boolean_question:
            extended_question = (
                f"Answer only yes or no, without any additional comments: {question}"
            )
        prompts = [
            Prompt(
                messages=[
                    Message(
                        text=extended_question,
                        image=image_path,
                        role="user",
                    )
                ],
                temperature=0.0,
            )
            for image_path in image_paths
        ]
        coroutines = [
            self.skip_exception(
                self.llm.invoke_with_runtime_and_cost(
                    prompt=prompt,
                    logger=logger,
                ),
                logger=logger,
            )
            for prompt in prompts
        ]
        responses = await asyncio.gather(*coroutines)
        result = [
            VisionModelOutputItem(
                image_path=image_path,
                response=response,
                log_odds=0.0,
                runtime=runtime,
                cost=costs,
            )
            for (image_path, (response, runtime, costs)) in zip(image_paths, responses)
        ]
        return result

    async def skip_exception(self, coroutine, logger):
        try:
            return await coroutine
        except Exception as e:
            logger.warning(
                __name__, f"Error occured during VLM {self.model_id} invocation {e}."
            )
            return ("Not sure", 0.0, 0.0)
