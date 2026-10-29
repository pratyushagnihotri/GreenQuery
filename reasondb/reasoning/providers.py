"""Multi-provider LLM support driven by the GreenQuery model registry.

``config/models.yaml`` lists every selectable model. ``make_api_llm`` turns a
registry key into a ReasonDB ``LargeLanguageModel`` for OpenAI, Anthropic or an
OpenAI-compatible local server (vLLM, Ollama, LM Studio). Unknown model ids fall
back to OpenAI so existing runner invocations keep working unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reasondb.reasoning.llm import (
    ConfigurableOpenAIModel,
    LargeLanguageModel,
    LLMCharacteristics,
    OpenAILLM,
    Prompt,
)

REGISTRY_PATH = Path(
    os.environ.get(
        "GREENQUERY_MODEL_REGISTRY",
        Path(__file__).parents[2] / "config" / "models.yaml",
    )
)

ENERGY_EXACT = "ecologits"
ENERGY_PROXY = "ecologits_proxy"
ENERGY_NONE = "not_available"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    api_model: str
    display_name: str
    api_key_env: Optional[str]
    base_url: Optional[str]
    price_in: float
    price_out: float
    energy: str
    ecologits_model: Optional[str]
    effort: Optional[str]
    max_tokens: int
    rpm: int
    tpm: int
    gpus: int = 1

    @property
    def credential_available(self) -> bool:
        return not self.api_key_env or bool(os.environ.get(self.api_key_env))

    @property
    def requires_ecologits(self) -> bool:
        return self.energy in (ENERGY_EXACT, ENERGY_PROXY)

    @property
    def deployment(self) -> str:
        return "Local server" if self.provider == "openai_compatible" else "Online API"

    def characteristics(self) -> LLMCharacteristics:
        return LLMCharacteristics(
            rpm=self.rpm,
            tpm=self.tpm,
            out_len=self.max_tokens,
            in_len=1_000_000,
            in_cost=self.price_in,
            out_cost=self.price_out,
        )

    def to_public_dict(self) -> Dict[str, Any]:
        """Catalog entry safe to send to the browser (no secrets)."""
        return {
            "key": self.key,
            "provider": self.provider,
            "api_model": self.api_model,
            "display_name": self.display_name,
            "deployment": self.deployment,
            "price_in": self.price_in,
            "price_out": self.price_out,
            "energy": self.energy,
            "ecologits_model": self.ecologits_model,
            "credential_env": self.api_key_env,
            "credential_available": self.credential_available,
        }


def _spec_from_entry(key: str, entry: Dict[str, Any]) -> ModelSpec:
    provider = entry.get("provider", "openai")
    if provider not in ("openai", "anthropic", "openai_compatible"):
        raise ValueError(f"Model {key}: unknown provider {provider!r}")
    base_url = entry.get("base_url")
    if entry.get("base_url_env") and os.environ.get(entry["base_url_env"]):
        base_url = os.environ[entry["base_url_env"]]
    energy = entry.get("energy", ENERGY_EXACT if provider != "openai_compatible" else ENERGY_NONE)
    if energy == ENERGY_PROXY and not entry.get("ecologits_model"):
        raise ValueError(f"Model {key}: energy=ecologits_proxy requires ecologits_model")
    return ModelSpec(
        key=key,
        provider=provider,
        api_model=entry.get("api_model", key),
        display_name=entry.get("display_name", key),
        api_key_env=entry.get("api_key_env"),
        base_url=base_url,
        price_in=float(entry.get("price_in", 0.0)),
        price_out=float(entry.get("price_out", 0.0)),
        energy=energy,
        ecologits_model=entry.get("ecologits_model"),
        effort=entry.get("effort"),
        max_tokens=int(entry.get("max_tokens", 4096)),
        rpm=int(entry.get("rpm", 500)),
        tpm=int(entry.get("tpm", 200_000)),
        gpus=int(entry.get("gpus", 1)),
    )


# Optional overlay with models added at runtime (GreenQuery dashboard: "bring your own model").
CUSTOM_REGISTRY_PATH = Path(os.environ.get("GREENQUERY_CUSTOM_MODELS", REGISTRY_PATH.parent / "models.custom.yaml"))


def load_custom_registry() -> Dict[str, dict]:
    import yaml

    if not CUSTOM_REGISTRY_PATH.exists():
        return {}
    data = yaml.safe_load(CUSTOM_REGISTRY_PATH.read_text()) or {}
    return data.get("models") or {}


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, ModelSpec]:
    import yaml

    data = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
    entries = {**load_custom_registry(), **(data.get("models") or {})}  # curated entries win
    return {key: _spec_from_entry(key, entry or {}) for key, entry in entries.items()}


def reload_registry() -> None:
    """Drop the cached registry (after the custom overlay changed)."""
    load_registry.cache_clear()


def get_model_spec(model_id: str) -> Optional[ModelSpec]:
    registry = load_registry()
    if model_id in registry:
        return registry[model_id]
    for spec in registry.values():
        if spec.api_model == model_id:
            return spec
    return None


def provider_for(model_id: str) -> str:
    spec = get_model_spec(model_id)
    return spec.provider if spec else "openai"


def model_requires_ecologits(model_id: str) -> bool:
    spec = get_model_spec(model_id)
    return spec.requires_ecologits if spec else True


def require_model_credentials(model_id: str) -> None:
    """Fail early, before any measurement starts, if the model's key is missing."""
    spec = get_model_spec(model_id)
    env = spec.api_key_env if spec else "OPENAI_API_KEY"
    if env and not os.environ.get(env):
        raise RuntimeError(f"{env} is required for model {model_id!r}")


def ecologits_estimate(
    provider: str, model_name: str, output_tokens: int, latency_seconds: float
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """EcoLogits impact for one call: energy kWh (mid, min, max), CO2 g (mid, min, max)."""
    try:
        from ecologits.tracers.utils import llm_impacts
    except ImportError:
        return None, None, None, None, None, None
    impacts = llm_impacts(
        provider=provider,
        model_name=model_name,
        output_token_count=output_tokens,
        request_latency=latency_seconds,
    )
    if impacts is None or getattr(impacts, "has_errors", False):
        return None, None, None, None, None, None

    def span(value, multiplier=1.0):
        raw = getattr(value, "value", value)
        low, high = getattr(raw, "min", raw), getattr(raw, "max", raw)
        low, high = float(low) * multiplier, float(high) * multiplier
        return (low + high) / 2, low, high

    energy_mid, energy_min, energy_max = span(impacts.energy)
    # EcoLogits reports GWP in kgCO2eq; GreenQuery records grams.
    co2_mid, co2_min, co2_max = span(impacts.gwp, 1_000.0)
    return energy_mid, energy_min, energy_max, co2_mid, co2_min, co2_max


def _record_call(
    spec: ModelSpec,
    *,
    resolved_model: str,
    runtime: float,
    in_tokens: int,
    out_tokens: int,
    in_cost: float,
    out_cost: float,
) -> None:
    from evaluation.measurement.api_usage import record_api_call

    energy = (None,) * 6
    source = ENERGY_NONE
    if spec.energy == ENERGY_EXACT:
        energy = ecologits_estimate(spec.provider, spec.api_model, out_tokens, runtime)
        source = "estimated_by_ecologits"
    elif spec.energy == ENERGY_PROXY:
        energy = ecologits_estimate(spec.provider, spec.ecologits_model or spec.api_model, out_tokens, runtime)
        # Same source label as exact estimates (validators rely on it); the proxy
        # model is kept in `ecologits_model` on every event.
        source = "estimated_by_ecologits"
    energy_kwh, energy_min, energy_max, co2_g, co2_min, co2_max = energy
    record_api_call(
        provider=spec.provider,
        model_name=resolved_model,
        requested_model_name=spec.api_model,
        resolved_model_name=resolved_model,
        runtime_seconds=runtime,
        prompt_tokens=in_tokens,
        completion_tokens=out_tokens,
        prompt_cost_usd=in_cost,
        completion_cost_usd=out_cost,
        energy_kwh=energy_kwh,
        energy_kwh_min=energy_min,
        energy_kwh_max=energy_max,
        co2_g=co2_g,
        co2_g_min=co2_min,
        co2_g_max=co2_max,
        energy_source=source,
        co2_source=source,
        ecologits_model=spec.ecologits_model if spec.energy == ENERGY_PROXY else None,
    )


class AnthropicLLM(LargeLanguageModel):
    """Claude via the Anthropic Messages API.

    Current Claude models reject non-default sampling parameters, so the prompt's
    temperature/seed are not forwarded. Refusals raise instead of falling back to
    another model: a silent model switch would invalidate the benchmark record.
    """

    def __init__(self, spec: ModelSpec, cache_enabled: bool):
        super().__init__(spec.api_model, spec.characteristics(), cache_enabled)
        self.spec = spec
        self.client = None

    async def prepare(self):
        await super().prepare()
        import anthropic

        api_key = os.environ.get(self.spec.api_key_env) if self.spec.api_key_env else None
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=600.0, max_retries=3)

    async def close(self):
        if self.client is not None:
            await self.client.close()
            self.client = None

    @staticmethod
    def _convert(prompt: Prompt) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system_parts: List[str] = []
        messages: List[Dict[str, Any]] = []
        for message in prompt.messages:
            if message.role == "system":
                system_parts.append(message.only_text)
                continue
            if message.image is not None:
                content: Any = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": prompt.encode_image(message.image),
                        },
                    },
                    {"type": "text", "text": message.only_text},
                ]
            else:
                content = message.only_text
            if messages and messages[-1]["role"] == message.role and isinstance(content, str) and isinstance(messages[-1]["content"], str):
                # The Messages API expects alternating turns; merge consecutive same-role text.
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": message.role, "content": content})
        return ("\n\n".join(system_parts) or None), messages

    async def _invoke(self, prompt: Prompt, stop: List[str] = []) -> Tuple[str, float, float]:
        assert self.client is not None, "Prepare LLM first"
        system, messages = self._convert(prompt)
        kwargs: Dict[str, Any] = {
            "model": self.spec.api_model,
            "max_tokens": self.spec.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        if self.spec.effort:
            kwargs["output_config"] = {"effort": self.spec.effort}
        start = time.time()
        response = await self.client.messages.create(**kwargs)
        runtime = time.time() - start
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{self.spec.api_model} refused the request: {response.stop_details}")
        text = "".join(block.text for block in response.content if block.type == "text")
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        in_cost = in_tokens / 1_000_000 * self.spec.price_in
        out_cost = out_tokens / 1_000_000 * self.spec.price_out
        _record_call(
            self.spec,
            resolved_model=response.model or self.spec.api_model,
            runtime=runtime,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            in_cost=in_cost,
            out_cost=out_cost,
        )
        return text, runtime, in_cost + out_cost


class OpenAICompatibleLLM(OpenAILLM):
    """Any server that speaks the OpenAI chat-completions protocol (vLLM, Ollama, ...)."""

    def __init__(self, spec: ModelSpec, cache_enabled: bool):
        # Skip OpenAILLM.__init__: it requires OPENAI_API_KEY.
        LargeLanguageModel.__init__(self, spec.api_model, spec.characteristics(), cache_enabled)
        self.spec = spec
        self.api_key = (os.environ.get(spec.api_key_env) if spec.api_key_env else None) or "not-needed"
        self.client = None

    async def prepare(self):
        # Skip OpenAILLM.prepare: it builds a client for api.openai.com.
        await LargeLanguageModel.prepare(self)
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.spec.base_url, timeout=600.0, max_retries=3)

    async def _invoke(self, prompt: Prompt, stop: List[str] = []) -> Tuple[str, float, float]:
        assert self.client is not None, "Prepare LLM first"
        start = time.time()
        response = await self.client.chat.completions.create(
            messages=prompt.prompt_messages,
            model=self.spec.api_model,
            temperature=prompt.temperature,
            stop=stop or None,
        )
        runtime = time.time() - start
        text = response.choices[0].message.content or ""
        usage = response.usage
        in_tokens = usage.prompt_tokens if usage else 0
        out_tokens = usage.completion_tokens if usage else 0
        in_cost = in_tokens / 1_000_000 * self.spec.price_in
        out_cost = out_tokens / 1_000_000 * self.spec.price_out
        _record_call(
            self.spec,
            resolved_model=response.model or self.spec.api_model,
            runtime=runtime,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            in_cost=in_cost,
            out_cost=out_cost,
        )
        return text, runtime, in_cost + out_cost


def make_api_llm(model_id: str, cache_enabled: bool = True) -> LargeLanguageModel:
    """Build the planner/executor LLM for a registry key or raw model id."""
    spec = get_model_spec(model_id)
    if spec is None or spec.provider == "openai":
        return ConfigurableOpenAIModel(spec.api_model if spec else model_id, cache_enabled=cache_enabled)
    if spec.provider == "anthropic":
        return AnthropicLLM(spec, cache_enabled)
    return OpenAICompatibleLLM(spec, cache_enabled)
