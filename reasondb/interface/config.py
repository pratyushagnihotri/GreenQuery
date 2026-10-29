import os

import torch
from reasondb.database.database import Database
from reasondb.executor import Executor
from reasondb.operators.extract.image_qa_extract import ImageQaExtract
from reasondb.operators.join.qa_filter_join import QaFilterJoin
from reasondb.operators.limit.limit import Limit
from reasondb.optimizer.baselines.abacus_optimizer import ParetoCascades
from reasondb.optimizer.baselines.lotus_optimizer import LotusOptimizer
from reasondb.optimizer.gd_optimizer import (
    GlobalOptimizationMode,
    GradientDescentOptimizer,
    OptimizationConfig,
)
from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
from reasondb.reasoning.llm import LocalTextServerModel
from reasondb.reasoning.providers import make_api_llm
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
from reasondb.backends.image_qa import VisionModelImageQABackend
from reasondb.backends.audio_qa import AudioModelAudioQABackend
from reasondb.backends.image_similarity import ImageSimilarityBackend
from reasondb.backends.python_codegen import (
    LLMPythonCodegenBackend,
)
from reasondb.backends.text_qa import KvTextQABackend, LLMTextQABackend
from reasondb.backends.vision_model import KvVisionModel, LlmVisionModel
from reasondb.backends.audio_model import KvAudioModel
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.extract.python_extract import PythonExtract
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.operators.filter.extract_and_match import (
    ExtractAndMatchFilter,
)
from reasondb.operators.filter.extract_and_qa_filter import ExtractAndQaFilter
from reasondb.operators.filter.extract_and_match_image import ExtractAndMatchImageFilter
from reasondb.operators.filter.extract_and_qa_image import ExtractAndQaImageFilter
from reasondb.operators.filter.image_qa_join_filter import ImageQaJoinFilter
from reasondb.operators.filter.raw_text_qa_filter import RawTextQaFilter
from reasondb.operators.filter.audio_qa_filter import AudioQaFilter
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.operators.join.traditional_join import TraditionalJoin
from reasondb.operators.project.project import Project
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.rename.rename import Rename
from reasondb.operators.sorting.sort import Sort
from reasondb.operators.tranform.python_transform import PythonTransform
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.physical_operator import CostType, FilterMode, PhysicalOperatorToolbox
from reasondb.backends.text_embeddings import TextSimilarityBackend


LOCAL_TEXT_BACKEND_PROFILES = {
    "dgx-local-text",
    "cloudlab-local-text",
    "dgx-local-text-vision",
    "dgx-local-text-vision-audio",
}


def operator_model_ids() -> list[str]:
    """Models offered as alternative semantic operators (multi-model mode).

    GREENQUERY_OPERATOR_MODELS is a comma-separated list ordered from cheapest to
    most capable, e.g. "ollama/qwen2.5:3b-instruct,ollama/qwen2.5:32b-instruct".
    Unset means single-model mode: one model plans and executes every operator.
    """
    raw = os.environ.get("GREENQUERY_OPERATOR_MODELS", "")
    return [model_id.strip() for model_id in raw.split(",") if model_id.strip()]


def apply_operator_models(
    configurator: PlanConfigurator, model_ids: list[str], cache_enabled: bool, base_cost: float
) -> None:
    """Offer each text-semantic operator once per model so optimizers have real choices.

    Quality and nominal cost rise along the list; the cheapest model's operators
    become Lotus proxies (``configurator.multi_model_proxies``).
    """
    toolbox = configurator.physical_operators
    count = len(model_ids)
    filters, extracts, join_predicates, proxies = [], [], [], []
    for rank, model_id in enumerate(model_ids):
        backend = LLMTextQABackend(make_api_llm(model_id, cache_enabled=cache_enabled))
        quality = 10 - (count - 1 - rank)
        cost = base_cost * (rank + 1) / count
        operators = (
            TextQaFilter(backend, quality=quality, fake_cost=cost),
            TextQaExtract(backend, quality=quality, fake_cost=cost),
            ExtractAndQaFilter(backend, fake_cost=cost, quality=quality),
        )
        filters.append(operators[0])
        extracts.append(operators[1])
        join_predicates.append(operators[2])
        if rank == 0:
            proxies = [operator.get_operation_identifier() for operator in operators]
    toolbox.filter_operators = [
        op for op in toolbox.filter_operators if not isinstance(op, TextQaFilter)
    ] + filters
    toolbox.extract_operators = [
        op for op in toolbox.extract_operators if not isinstance(op, TextQaExtract)
    ] + extracts
    toolbox.join_predicates = [
        op for op in toolbox.join_predicates if not isinstance(op, ExtractAndQaFilter)
    ] + join_predicates
    configurator.multi_model_proxies = proxies


def get_default_configurator(
    backend_profile: str = "api",
    local_cache_policy: str = "default",
    openai_model_id: str | None = None,
    openai_cache_enabled: bool = True,
    local_text_model_id: str | None = None,
    local_planner_model_id: str | None = None,
):
    if backend_profile not in ("api", *LOCAL_TEXT_BACKEND_PROFILES):
        raise ValueError(f"Unknown backend profile: {backend_profile}")
    if local_cache_policy not in ("default", "disabled"):
        raise ValueError(f"Unknown local cache policy: {local_cache_policy}")

    # Original local-model cost presets kept for reference.
    # kv099_cost = 0.05
    # kv09_cost = 0.1
    # kv08_cost = 0.3
    # kv06_cost = 0.45
    # kv05_cost = 0.6
    # kv04_cost = 0.75
    # kv03_cost = 0.85
    # kv00_cost = 0.9
    gpt_cost = 2
    if local_planner_model_id:
        planner_model = LocalTextServerModel(local_planner_model_id, cache_enabled=False)
        api_text_model = planner_model
    else:
        api_text_model = make_api_llm(
            openai_model_id or os.environ.get("GREENQUERY_OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
            cache_enabled=openai_cache_enabled,
        )
        planner_model = make_api_llm(
            openai_model_id or os.environ.get("GREENQUERY_OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
            cache_enabled=openai_cache_enabled,
        )
    textqa_backend = LLMTextQABackend(api_text_model)
    # Disabled per Matthias: local embedding backend.
    # text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    imageqa_backend = VisionModelImageQABackend(LlmVisionModel(api_text_model))

    local_text_similarity_backend = None
    local_textqa_backends = {}
    local_vanilla = os.environ.get("GREENQUERY_LOCAL_VANILLA") == "1"
    if backend_profile in LOCAL_TEXT_BACKEND_PROFILES:
        local_text_model_id = local_text_model_id or os.environ.get(
            "GREENQUERY_LOCAL_TEXT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
        )
        kv08_cost = 0.3
        kv05_cost = 0.6
        kv00_cost = 0.9
        local_text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
        if local_vanilla:
            local_textqa_backends = {
                "cr00": KvTextQABackend(
                    local_text_model_id,
                    compression_ratio=0.0,
                    cache_enabled=local_cache_policy != "disabled",
                )
            }
        else:
            local_textqa_backends = {
                "cr08": KvTextQABackend(
                    local_text_model_id,
                    compression_ratio=0.8,
                    cache_enabled=local_cache_policy != "disabled",
                ),
                "cr05": KvTextQABackend(
                    local_text_model_id,
                    compression_ratio=0.5,
                    cache_enabled=local_cache_policy != "disabled",
                ),
                "cr00": KvTextQABackend(
                    local_text_model_id,
                    compression_ratio=0.0,
                    cache_enabled=local_cache_policy != "disabled",
                ),
            }
        if backend_profile in {"dgx-local-text-vision", "dgx-local-text-vision-audio"}:
            vision_model_id = os.environ.get(
                "GREENQUERY_LOCAL_VISION_MODEL", "llava-hf/llama3-llava-next-8b-hf"
            )
            imageqa_backend = VisionModelImageQABackend(
                KvVisionModel(
                    vision_model_id,
                    compression_ratio=0.0,
                    cache_enabled=local_cache_policy != "disabled",
                )
            )
    # Disabled per Matthias: all local KV text backends.
    # kvtextqa8B_backendcr09 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.9
    # )
    # kvtextqa8B_backendcr08 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.8
    # )
    # kvtextqa8B_backendcr06 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.6
    # )
    # kvtextqa8B_backendcr05 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.5
    # )
    # kvtextqa8B_backendcr04 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.4
    # )
    # kvtextqa8B_backendcr03 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.3
    # )
    # kvtextqa8B_backendcr00 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.0
    # )
    # kvtextqa70B_backendcr09 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.9
    # )
    # kvtextqa70B_backendcr08 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.8
    # )
    # kvtextqa70B_backendcr06 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.6
    # )
    # kvtextqa70B_backendcr05 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.5
    # )
    # kvtextqa70B_backendcr04 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.4
    # )
    # kvtextqa70B_backendcr03 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.3
    # )
    # kvtextqa70B_backendcr00 = KvTextQABackend(
    #     "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.0
    # )
    # Disabled per Matthias: all local KV image backends.
    # kvimageqa8B_backendcr0995 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.995)
    # )
    # kvimageqa8B_backendcr099 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.99)
    # )
    # kvimageqa8B_backendcr095 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.95)
    # )
    # kvimageqa8B_backendcr09 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.9)
    # )
    # kvimageqa8B_backendcr08 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.8)
    # )
    # kvimageqa8B_backendcr05 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.5)
    # )
    # kvimageqa8B_backendcr00 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.0)
    # )
    # kvimageqa70B_backendcr0995 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.995)
    # )
    # kvimageqa70B_backendcr099 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.99)
    # )
    # kvimageqa70B_backendcr095 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.95)
    # )
    # kvimageqa70B_backendcr09 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.9)
    # )
    # kvimageqa70B_backendcr08 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.8)
    # )
    # kvimageqa70B_backendcr05 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.5)
    # )
    # kvimageqa70B_backendcr00 = VisionModelImageQABackend(
    #     KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.0)
    # )
    # Disabled per Matthias: all local KV audio backends.
    # kvaudioqa_backend_cr09 = AudioModelAudioQABackend(
    #     KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.9)
    # )
    # kvaudioqa_backend_cr08 = AudioModelAudioQABackend(
    #     KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.8)
    # )
    # kvaudioqa_backend_cr05 = AudioModelAudioQABackend(
    #     KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.5)
    # )
    # kvaudioqa_backend_cr00 = AudioModelAudioQABackend(
    #     KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.0)
    # )
    configurator = PlanConfigurator(
        llm=planner_model,
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                QaFilterJoin(),
            ],
            join_predicates=[
                # Disabled local-model join predicates kept for reference.
                # ExtractAndMatchFilter(
                #     kvtextqa8B_backendcr08,
                #     text_similarity_backend,
                #     fake_cost=kv08_cost,
                #     quality=1,
                # ),
                # ExtractAndMatchFilter(
                #     kvtextqa8B_backendcr05,
                #     text_similarity_backend,
                #     fake_cost=kv05_cost,
                #     quality=1.3,
                # ),
                # ExtractAndMatchFilter(
                #     kvtextqa8B_backendcr00,
                #     text_similarity_backend,
                #     fake_cost=kv00_cost,
                #     quality=1.7,
                # ),
                # ExtractAndMatchFilter(
                #     kvtextqa70B_backendcr08,
                #     text_similarity_backend,
                #     fake_cost=kv08_cost,
                #     quality=2,
                # ),
                # ExtractAndMatchFilter(
                #     kvtextqa70B_backendcr06,
                #     text_similarity_backend,
                #     fake_cost=kv06_cost,
                #     quality=2.3,
                # ),
                # ExtractAndMatchFilter(
                #     kvtextqa70B_backendcr03,
                #     text_similarity_backend,
                #     fake_cost=kv03_cost,
                #     quality=2.7,
                # ),
                # ExtractAndMatchFilter(
                #     kvtextqa70B_backendcr00,
                #     text_similarity_backend,
                #     fake_cost=kv00_cost,
                #     quality=2.9,
                # ),
                ExtractAndQaFilter(
                    textqa_backend,
                    fake_cost=gpt_cost,
                    quality=10,
                ),
                # ExtractAndQaFilter(
                #     kvtextqa8B_backendcr08,
                #     fake_cost=kv08_cost,
                #     quality=3,
                # ),
                # ExtractAndQaFilter(
                #     kvtextqa8B_backendcr05,
                #     fake_cost=kv05_cost,
                #     quality=3.3,
                # ),
                # ExtractAndQaFilter(
                #     kvtextqa8B_backendcr00,
                #     fake_cost=kv00_cost,
                #     quality=3.7,
                # ),
                # ExtractAndQaFilter(
                #     kvtextqa70B_backendcr08,
                #     fake_cost=kv08_cost,
                #     quality=4,
                # ),
                # ExtractAndQaFilter(
                #     kvtextqa70B_backendcr06,
                #     fake_cost=kv06_cost,
                #     quality=4.1,
                # ),
                # ExtractAndQaFilter(
                #     kvtextqa70B_backendcr03,
                #     fake_cost=kv03_cost,
                #     quality=4.3,
                # ),
                # ExtractAndQaFilter(
                #     kvtextqa70B_backendcr00,
                #     fake_cost=kv00_cost,
                #     quality=4.7,
                # ),
                # ExtractAndMatchImageFilter(
                #     kvimageqa8B_backendcr09,
                #     text_similarity_backend,
                #     fake_cost=kv09_cost,
                #     quality=1,
                # ),
                # ExtractAndMatchImageFilter(
                #     kvimageqa8B_backendcr05,
                #     text_similarity_backend,
                #     fake_cost=kv05_cost,
                #     quality=1.3,
                # ),
                # ExtractAndMatchImageFilter(
                #     kvimageqa8B_backendcr00,
                #     text_similarity_backend,
                #     fake_cost=kv00_cost,
                #     quality=1.7,
                # ),
                # ExtractAndMatchImageFilter(
                #     kvimageqa70B_backendcr099,
                #     text_similarity_backend,
                #     fake_cost=kv099_cost,
                #     quality=2,
                # ),
                # ExtractAndMatchImageFilter(
                #     kvimageqa70B_backendcr09,
                #     text_similarity_backend,
                #     fake_cost=kv09_cost,
                #     quality=2.3,
                # ),
                # ExtractAndMatchImageFilter(
                #     kvimageqa70B_backendcr05,
                #     text_similarity_backend,
                #     fake_cost=kv05_cost,
                #     quality=2.7,
                # ),
                # ExtractAndMatchImageFilter(
                #     kvimageqa70B_backendcr00,
                #     text_similarity_backend,
                #     fake_cost=kv00_cost,
                #     quality=2.9,
                # ),
                ExtractAndQaImageFilter(
                    imageqa_backend,
                    fake_cost=gpt_cost,
                    quality=10,
                ),
                # ExtractAndQaImageFilter(
                #     kvimageqa8B_backendcr09,
                #     fake_cost=kv09_cost,
                #     quality=3,
                # ),
                # ExtractAndQaImageFilter(
                #     kvimageqa8B_backendcr05,
                #     fake_cost=kv05_cost,
                #     quality=3.3,
                # ),
                # ExtractAndQaImageFilter(
                #     kvimageqa8B_backendcr00,
                #     fake_cost=kv00_cost,
                #     quality=3.7,
                # ),
                # ExtractAndQaImageFilter(
                #     kvimageqa70B_backendcr099,
                #     fake_cost=kv099_cost,
                #     quality=4,
                # ),
                # ExtractAndQaImageFilter(
                #     kvimageqa70B_backendcr09,
                #     fake_cost=kv09_cost,
                #     quality=4.1,
                # ),
                # ExtractAndQaImageFilter(
                #     kvimageqa70B_backendcr05,
                #     fake_cost=kv05_cost,
                #     quality=4.3,
                # ),
                # ExtractAndQaImageFilter(
                #     kvimageqa70B_backendcr00,
                #     fake_cost=kv00_cost,
                #     quality=4.7,
                # ),
            ],
            filter_operators=[
                # Disabled local-model filter operators kept for reference.
                # TextQaFilter(textqa_backend, quality=10, fake_cost=gpt_cost),
                TraditionalFilter(quality=1, fake_cost=0),
                # ImageSimilarityFilter(
                #     ImageSimilarityBackend("Salesforce/blip-itm-base-coco"),
                #     quality=2,
                #     fake_cost=0,
                # ),
                # TextQaFilter(kvtextqa8B_backendcr09, quality=3, fake_cost=kv09_cost),
                # TextQaFilter(kvtextqa8B_backendcr08, quality=3.5, fake_cost=kv08_cost),
                # TextQaFilter(kvtextqa8B_backendcr06, quality=4, fake_cost=kv06_cost),
                # TextQaFilter(kvtextqa8B_backendcr05, quality=4.2, fake_cost=kv05_cost),
                # TextQaFilter(kvtextqa8B_backendcr04, quality=4.5, fake_cost=kv04_cost),
                # TextQaFilter(kvtextqa8B_backendcr03, quality=4.7, fake_cost=kv03_cost),
                # TextQaFilter(kvtextqa8B_backendcr00, quality=5, fake_cost=kv00_cost),
                # TextQaFilter(kvtextqa70B_backendcr09, quality=6, fake_cost=kv09_cost),
                # TextQaFilter(kvtextqa70B_backendcr08, quality=6.5, fake_cost=kv08_cost),
                # TextQaFilter(kvtextqa70B_backendcr06, quality=7, fake_cost=kv06_cost),
                # TextQaFilter(kvtextqa70B_backendcr05, quality=7.2, fake_cost=kv05_cost),
                # TextQaFilter(kvtextqa70B_backendcr04, quality=7.5, fake_cost=kv04_cost),
                # TextQaFilter(kvtextqa70B_backendcr03, quality=7.7, fake_cost=kv03_cost),
                # TextQaFilter(kvtextqa70B_backendcr00, quality=8, fake_cost=kv00_cost),
                TextQaFilter(textqa_backend, quality=10, fake_cost=gpt_cost),
                # ImageQaFilter(
                #     VisionModelImageQABackend(
                #         LocalVisionModel("Salesforce/blip2-opt-2.7b")
                #     )
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr0995,
                #     quality=2,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr099,
                #     quality=2.5,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr095,
                #     quality=2.7,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr09,
                #     quality=3,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr08,
                #     quality=4,
                #     fake_cost=kv08_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr05,
                #     quality=4.5,
                #     fake_cost=kv05_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr00,
                #     quality=5,
                #     fake_cost=kv00_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr0995,
                #     quality=2,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr099,
                #     quality=6,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr095,
                #     quality=2.7,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr09,
                #     quality=7,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr08,
                #     quality=4,
                #     fake_cost=kv08_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr05,
                #     quality=8,
                #     fake_cost=kv05_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr00,
                #     quality=9,
                #     fake_cost=kv00_cost,
                # ),
                ImageQaFilter(imageqa_backend, quality=11, fake_cost=gpt_cost),
                # AudioQaFilter(
                #     kvaudioqa_backend_cr09,
                #     quality=3,
                #     fake_cost=kv09_cost,
                # ),
                # AudioQaFilter(
                #     kvaudioqa_backend_cr08,
                #     quality=4,
                #     fake_cost=kv08_cost,
                # ),
                # AudioQaFilter(
                #     kvaudioqa_backend_cr05,
                #     quality=4.5,
                #     fake_cost=kv05_cost,
                # ),
                # AudioQaFilter(
                #     kvaudioqa_backend_cr00,
                #     quality=5,
                #     fake_cost=kv00_cost,
                # ),
            ],
            extract_operators=[
                # Disabled local-model extract operators kept for reference.
                # TextQaExtract(kvtextqa8B_backendcr09, quality=3, fake_cost=kv09_cost),
                # TextQaExtract(kvtextqa8B_backendcr08, quality=3.5, fake_cost=kv08_cost),
                # TextQaExtract(kvtextqa8B_backendcr06, quality=4, fake_cost=kv06_cost),
                # TextQaExtract(kvtextqa8B_backendcr05, quality=4.2, fake_cost=kv05_cost),
                # TextQaExtract(
                #     kvtextqa8B_backendcr04, quality=4.5, fake_cost=kv04_cost
                # ),
                # TextQaExtract(
                #     kvtextqa8B_backendcr03, quality=4.7, fake_cost=kv03_cost
                # ),
                # TextQaExtract(kvtextqa8B_backendcr00, quality=5, fake_cost=kv00_cost),
                # TextQaExtract(kvtextqa70B_backendcr09, quality=6, fake_cost=kv09_cost),
                # TextQaExtract(
                #     kvtextqa70B_backendcr08, quality=6.5, fake_cost=kv08_cost
                # ),
                # TextQaExtract(kvtextqa70B_backendcr06, quality=7, fake_cost=kv06_cost),
                # TextQaExtract(
                #     kvtextqa70B_backendcr05, quality=7.2, fake_cost=kv05_cost
                # ),
                # TextQaExtract(
                #     kvtextqa70B_backendcr04, quality=7.5, fake_cost=kv04_cost
                # ),
                # TextQaExtract(
                #     kvtextqa70B_backendcr03, quality=7.7, fake_cost=kv03_cost
                # ),
                # TextQaExtract(kvtextqa70B_backendcr00, quality=8, fake_cost=kv00_cost),
                TextQaExtract(textqa_backend, quality=10, fake_cost=gpt_cost),
                # ImageQaExtract(
                #     kvimageqa8B_backendcr09,
                #     quality=3,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaExtract(
                #     kvimageqa8B_backendcr05,
                #     quality=4.5,
                #     fake_cost=kv05_cost,
                # ),
                # ImageQaExtract(
                #     kvimageqa8B_backendcr00,
                #     quality=5,
                #     fake_cost=kv00_cost,
                # ),
                # ImageQaExtract(
                #     kvimageqa70B_backendcr099, quality=6, fake_cost=kv09_cost
                # ),
                # ImageQaExtract(
                #     kvimageqa70B_backendcr09, quality=7, fake_cost=kv09_cost
                # ),
                # ImageQaExtract(
                #     kvimageqa70B_backendcr05, quality=8, fake_cost=kv05_cost
                # ),
                # ImageQaExtract(
                #     kvimageqa70B_backendcr00, quality=9, fake_cost=kv00_cost
                # ),
                ImageQaExtract(imageqa_backend, quality=10, fake_cost=gpt_cost),
                PythonExtract(LLMPythonCodegenBackend(planner_model), quality=2, fake_cost=0),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(planner_model), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    model_ids = operator_model_ids()
    if model_ids:
        if backend_profile != "api":
            raise ValueError("GREENQUERY_OPERATOR_MODELS requires the api backend profile")
        apply_operator_models(configurator, model_ids, openai_cache_enabled, gpt_cost)
    if backend_profile in LOCAL_TEXT_BACKEND_PROFILES:
        toolbox = configurator.physical_operators
        kv00 = local_textqa_backends["cr00"]
        assert local_text_similarity_backend is not None
        if local_vanilla:
            toolbox.join_predicates = [
                ExtractAndMatchFilter(
                    kv00,
                    local_text_similarity_backend,
                    fake_cost=kv00_cost,
                    quality=1.7,
                ),
                ExtractAndQaFilter(kv00, fake_cost=kv00_cost, quality=3.7),
            ]
            toolbox.filter_operators = [
                TraditionalFilter(quality=1, fake_cost=0),
                TextQaFilter(kv00, quality=5, fake_cost=kv00_cost),
            ]
            toolbox.extract_operators = [
                TextQaExtract(kv00, quality=5, fake_cost=kv00_cost),
            ]
            return configurator

        kv08 = local_textqa_backends["cr08"]
        kv05 = local_textqa_backends["cr05"]
        toolbox.join_predicates = [
            ExtractAndMatchFilter(
                kv08,
                local_text_similarity_backend,
                fake_cost=kv08_cost,
                quality=1,
            ),
            ExtractAndMatchFilter(
                kv05,
                local_text_similarity_backend,
                fake_cost=kv05_cost,
                quality=1.3,
            ),
            ExtractAndMatchFilter(
                kv00,
                local_text_similarity_backend,
                fake_cost=kv00_cost,
                quality=1.7,
            ),
            ExtractAndQaFilter(kv08, fake_cost=kv08_cost, quality=3),
            ExtractAndQaFilter(kv05, fake_cost=kv05_cost, quality=3.3),
            ExtractAndQaFilter(kv00, fake_cost=kv00_cost, quality=3.7),
        ]
        toolbox.filter_operators = [
            TraditionalFilter(quality=1, fake_cost=0),
            TextQaFilter(kv08, quality=3.5, fake_cost=kv08_cost),
            TextQaFilter(kv05, quality=4.2, fake_cost=kv05_cost),
            TextQaFilter(kv00, quality=5, fake_cost=kv00_cost),
        ]
        toolbox.extract_operators = [
            TextQaExtract(kv08, quality=3.5, fake_cost=kv08_cost),
            TextQaExtract(kv05, quality=4.2, fake_cost=kv05_cost),
            TextQaExtract(kv00, quality=5, fake_cost=kv00_cost),
        ]
        if backend_profile in {"dgx-local-text-vision", "dgx-local-text-vision-audio"}:
            toolbox.filter_operators.append(ImageQaFilter(imageqa_backend, quality=10, fake_cost=gpt_cost))
            toolbox.extract_operators.append(ImageQaExtract(imageqa_backend, quality=10, fake_cost=gpt_cost))
        if backend_profile == "dgx-local-text-vision-audio":
            audio_model_id = os.environ.get("GREENQUERY_LOCAL_AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")
            audio_filter = AudioQaFilter(
                AudioModelAudioQABackend(KvAudioModel(
                    audio_model_id,
                    compression_ratio=0.0,
                    cache_enabled=local_cache_policy != "disabled",
                )),
                quality=10,
                fake_cost=gpt_cost,
            )
            audio_filter.set_mode(FilterMode.FILTER)
            audio_filter.register_other_operators(toolbox)
            toolbox.filter_operators.append(audio_filter)
    return configurator


def apply_operator_policy(configurator: PlanConfigurator, operator_policy: str):
    """Restrict the operator toolbox for benchmark-specific run modes."""
    if operator_policy == "default":
        return

    if operator_policy not in {"semantic-text-only", "semantic-multimodal"}:
        raise ValueError(f"Unknown operator policy: {operator_policy}")

    toolbox = configurator.physical_operators
    # Semantic text joins are rewritten into a database Cartesian join plus a
    # semantic join-predicate filter. Keep only the former here.
    toolbox.join_operators = [op for op in toolbox.join_operators if isinstance(op, TraditionalJoin)]
    toolbox.join_predicates = [
        op for op in toolbox.join_predicates if isinstance(op, ExtractAndQaFilter)
    ]
    toolbox.filter_operators = [
        op
        for op in toolbox.filter_operators
        if isinstance(op, (TraditionalFilter, TextQaFilter, ImageQaFilter, AudioQaFilter))
    ]
    toolbox.extract_operators = [
        op
        for op in toolbox.extract_operators
        if isinstance(op, (TextQaExtract, ImageQaExtract))
    ]
    if operator_policy == "semantic-text-only":
        toolbox.filter_operators = [op for op in toolbox.filter_operators if isinstance(op, TextQaFilter)]
        toolbox.extract_operators = [op for op in toolbox.extract_operators if isinstance(op, TextQaExtract)]


class Config:
    def __init__(
        self,
        identifier_for_caching: str,
        working_dir: str = "./.working_dir",
        device: torch.device = torch.device("cpu"),
        operator_policy: str = "default",
        backend_profile: str = "api",
        local_cache_policy: str = "default",
        openai_model_id: str | None = None,
        openai_cache_enabled: bool = True,
        local_text_model_id: str | None = None,
        local_planner_model_id: str | None = None,
    ):
        self.working_dir = working_dir
        self.db_file = None
        self.identifier_for_caching = identifier_for_caching
        self.device = device
        self.operator_policy = operator_policy
        self.backend_profile = backend_profile
        self.local_cache_policy = local_cache_policy
        self.openai_model_id = openai_model_id
        self.openai_cache_enabled = openai_cache_enabled
        self.local_text_model_id = local_text_model_id
        self.local_planner_model_id = local_planner_model_id

    def construct_executor(self):
        configurator = get_default_configurator(
            self.backend_profile,
            self.local_cache_policy,
            self.openai_model_id,
            self.openai_cache_enabled,
            self.local_text_model_id,
            self.local_planner_model_id,
        )
        api_model_id = self.openai_model_id or os.environ.get(
            "GREENQUERY_OPENAI_MODEL", "gpt-4o-mini-2024-07-18"
        )
        apply_operator_policy(configurator, self.operator_policy)
        planner_llm = (
            LocalTextServerModel(self.local_planner_model_id, cache_enabled=False)
            if self.local_planner_model_id
            else make_api_llm(api_model_id, cache_enabled=self.openai_cache_enabled)
        )
        reasoner = SelfCorrectionReasoner(
            llm=planner_llm,
            configurator=configurator,
            logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
            few_shot_database=DUMMY_FEW_SHOT_DATABASE,
        )
        database = Database(
            identifier_for_caching=self.identifier_for_caching,
            metadata_llm=(
                LocalTextServerModel(self.local_planner_model_id, cache_enabled=False)
                if self.local_planner_model_id
                else make_api_llm(api_model_id, cache_enabled=self.openai_cache_enabled)
            ),
            prepare_llm=(
                LocalTextServerModel(self.local_planner_model_id, cache_enabled=False)
                if self.local_planner_model_id
                else make_api_llm(api_model_id, cache_enabled=self.openai_cache_enabled)
            ),
        )
        gd_optimizer = GradientDescentOptimizer(
            OptimizationConfig(
                cost_type=CostType.RUNTIME,
                global_optimization_mode=GlobalOptimizationMode.COMBO,
                device=self.device,
            )
        )
        abacus_optimizer = ParetoCascades(CostType.RUNTIME)
        lotus_optimizer = LotusOptimizer(
            CostType.RUNTIME,
            # Multi-model mode: the cheapest configured model is the proxy.
            proxy_operators=getattr(configurator, "multi_model_proxies", None) or [
                        "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
                        "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
                        # "ImageSimilarityFilter-ImageSimilarityBackend-Salesforce/blip-itm-base-coco",
                        "AudioQaFilter-AudioQABackend-Qwen/Qwen2-Audio-7B-Instruct-cr0.9",  # cr00 is silver model
                        "ExtractAndMatchFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
                        "ExtractAndMatchImageFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
            ],
        )
        # GREENQUERY_OPTIMIZER selects the optimizer under test: stretto (default),
        # abacus (Palimpzest-style Pareto cascades) or lotus (Lotus-style proxies).
        optimizer_name = os.environ.get("GREENQUERY_OPTIMIZER", "stretto")
        optimizers = {"stretto": gd_optimizer, "abacus": abacus_optimizer, "lotus": lotus_optimizer}
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown GREENQUERY_OPTIMIZER: {optimizer_name}")
        executor = Executor(
            database=database,
            reasoner=reasoner,
            optimizer=optimizers[optimizer_name],
            configurator=configurator,
        )
        return executor
