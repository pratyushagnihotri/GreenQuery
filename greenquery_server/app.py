"""GreenQuery benchmark service.

Run with: ``uvicorn greenquery_server.app:app --port 8000`` from the repo root.
API keys are read from the server environment or the repo-root ``.env`` and are
never returned to clients.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from greenquery_server import settings
from greenquery_server.adapters import PLANNED, SYSTEMS, RunSpec
from greenquery_server.catalog import WORKLOADS
from greenquery_server.costs import estimate_cost
from greenquery_server.jobs import JobManager, new_job_id, spec_to_dict
from greenquery_server.live import live_progress
from evaluation.measurement.trace_summary import summarize_run_dir
from greenquery_server.store import Store
from reasondb.reasoning.providers import (CUSTOM_REGISTRY_PATH, get_model_spec, load_custom_registry, load_registry,
                                         reload_registry)

store = Store()
manager = JobManager(store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.import_archive()
    manager.start()
    yield
    await manager.stop()


app = FastAPI(title="GreenQuery benchmark service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class RunRequest(BaseModel):
    system: str = "stretto"
    workload: str
    model: str
    query: Optional[str] = None
    repetitions: int = Field(1, ge=1)
    cache_policy: Literal["disabled", "default"] = "disabled"
    # Multi-model mode: alternative operator models, cheapest first. None/empty = single-model.
    operator_models: Optional[List[str]] = None
    # Required when no cost estimate exists for a paid model.
    accept_unknown_cost: bool = False
    # Only estimate; do not queue.
    dry_run: bool = False


@app.get("/api/health")
def health():
    return {"status": "ok"}


def local_model_installed(spec) -> bool:
    """Whether a local (OpenAI-compatible) model can actually be served here."""
    if spec.provider != "openai_compatible" or spec.remote:
        return True
    if spec.key.startswith("ollama/"):
        if spec.key in load_custom_registry():
            return True  # added from the dashboard: the run job pulls it first
        manifests = settings.OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai" / "library"
        if not manifests.exists():
            return True  # no store to check (e.g. laptop with a running Ollama): assume yes
        name, _, tag = spec.api_model.partition(":")
        return (manifests / name / (tag or "latest")).exists()
    # vLLM and other servers: only when their URL is configured explicitly.
    return bool(os.environ.get("GREENQUERY_VLLM_BASE_URL"))


def public_model(spec) -> dict:
    data = spec.to_public_dict()
    if not local_model_installed(spec):
        data["credential_available"] = False
        data["credential_env"] = "not installed here"
    data["custom"] = spec.key in load_custom_registry()
    return data


# "Bring your own model". Sources:
#   ollama       any Ollama library model           -> runs on one GPU of the DGX, measured energy
#   huggingface  any GGUF repository on Hugging Face -> pulled by Ollama (hf.co/...), measured energy
#   openai       any OpenAI model id                 -> online API, through the logging proxy with a spending cap
#   openrouter   any OpenRouter model id             -> online API (Claude, Gemini, Llama, ...), same proxy and cap
# Online API models have no measured energy (inference runs at the provider).
OLLAMA_TAG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}(:[a-z0-9][a-z0-9._-]{0,63})?$")
HF_REPO = re.compile(r"^(?:hf\.co/|huggingface\.co/|https://huggingface\.co/)?([A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95})(?::([A-Za-z0-9._-]{1,32}))?/?$")
API_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
HF_DEFAULT_QUANT = "Q4_K_M"  # what Ollama pulls from a GGUF repository when no quantization is given


def _get_json(url: str, headers: Optional[dict] = None) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "GreenQuery", **(headers or {})})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def ollama_model_size(name: str, tag: str) -> int:
    """Download size in bytes from the Ollama registry manifest (raises LookupError if unknown)."""
    url = f"https://registry.ollama.ai/v2/library/{name}/manifests/{tag}"
    try:
        manifest = _get_json(url, {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    except urllib.error.HTTPError as error:
        raise LookupError(f"{name}:{tag} is not in the Ollama library") from error
    return sum(int(layer.get("size", 0)) for layer in manifest.get("layers", []))


def huggingface_gguf(repo: str, quant: Optional[str]) -> tuple:
    """(quantization, size in bytes) of a GGUF repository on Hugging Face (raises LookupError)."""
    try:
        info = _get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
    except urllib.error.HTTPError as error:
        raise LookupError(f"{repo} is not a public model on Hugging Face") from error
    files = [(item["rfilename"], int(item.get("size") or 0)) for item in info.get("siblings", [])
             if item.get("rfilename", "").lower().endswith(".gguf")]
    if not files:
        raise LookupError(f"{repo} has no GGUF files; pick a GGUF repository (e.g. bartowski/...-GGUF)")
    wanted = quant or HF_DEFAULT_QUANT
    matching = [(name, size) for name, size in files if wanted.lower() in name.lower()]
    if not matching:
        options = sorted({m.group(0).upper() for name, _ in files
                          for m in [re.search(r"(I?Q\d[\w]*|F16|BF16|F32)", name.rsplit("/", 1)[-1], re.I)] if m})
        raise LookupError(f"{repo} has no {wanted} file; add one of these after a colon: {', '.join(options) or 'none found'}")
    return wanted, sum(size for _, size in matching)


class CustomModel(BaseModel):
    source: Literal["ollama", "huggingface", "openai", "openrouter"] = "ollama"
    model: str = Field("", max_length=160)
    tag: str = Field("", max_length=160)  # older clients: Ollama tag
    price_in: Optional[float] = Field(None, ge=0)   # USD per million tokens (OpenAI: required)
    price_out: Optional[float] = Field(None, ge=0)


def _custom_local(key: str, api_model: str, display: str, size: int) -> dict:
    if size > settings.CUSTOM_MODEL_MAX_BYTES:
        raise HTTPException(400, f"{api_model} is {size / 1e9:.0f} GB; at most "
                                 f"{settings.CUSTOM_MODEL_MAX_BYTES / 1e9:.0f} GB fit on one GPU here")
    return {"provider": "openai_compatible", "api_model": api_model, "display_name": display,
            "base_url": "http://localhost:11434/v1", "base_url_env": "GREENQUERY_OLLAMA_BASE_URL",
            "price_in": 0.0, "price_out": 0.0, "energy": "not_available", "size_bytes": size}


def _custom_entry(body: CustomModel) -> tuple:
    """(registry key, registry entry) for a model added from the dashboard."""
    raw = (body.model or body.tag).strip()
    if not raw:
        raise HTTPException(400, "Enter a model name")
    if body.source == "ollama":
        tag = raw.lower()
        if not OLLAMA_TAG.match(tag):
            raise HTTPException(400, "Use an Ollama library name such as phi3:mini or mistral:7b")
        name, _, version = tag.partition(":")
        version = version or "latest"
        key = f"ollama/{name}:{version}"
        if get_model_spec(key) is not None:
            return key, None
        try:
            size = ollama_model_size(name, version)
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        return key, _custom_local(key, f"{name}:{version}", f"{name}:{version} (Ollama, added)", size)
    if body.source == "huggingface":
        match = HF_REPO.match(raw)
        if not match:
            raise HTTPException(400, "Use a Hugging Face repository such as bartowski/Phi-3.5-mini-instruct-GGUF "
                                     "(optionally :Q4_K_M)")
        repo, quant = match.group(1), match.group(2)
        try:
            quant, size = huggingface_gguf(repo, quant)
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        api_model = f"hf.co/{repo}:{quant}"
        key = f"ollama/{api_model}"
        if get_model_spec(key) is not None:
            return key, None
        return key, _custom_local(key, api_model, f"{repo.split('/')[-1]} {quant} (Hugging Face, added)", size)
    if not API_MODEL_ID.match(raw):
        raise HTTPException(400, "Use the provider's model id, e.g. gpt-4.1-mini or anthropic/claude-haiku-5.5")
    if body.source == "openai":
        key = f"openai/{raw}"
        if get_model_spec(key) is not None or get_model_spec(raw) is not None:
            return (key if get_model_spec(key) else get_model_spec(raw).key), None
        if not body.price_in or not body.price_out:
            raise HTTPException(400, "Enter the model's input and output price (USD per million tokens); "
                                     "the spending cap of every run depends on it")
        if os.environ.get("OPENAI_API_KEY"):  # check the id exists before anyone starts a run with it
            try:
                _get_json(f"https://api.openai.com/v1/models/{raw}",
                          {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
            except urllib.error.HTTPError as error:
                raise HTTPException(404, f"OpenAI does not offer {raw!r} to this account") from error
        return key, {"provider": "openai", "api_model": raw, "display_name": f"{raw} (OpenAI, added)",
                     "api_key_env": "OPENAI_API_KEY", "price_in": body.price_in, "price_out": body.price_out,
                     "energy": "not_available"}
    # OpenRouter: model list and prices are public.
    key = f"openrouter/{raw}"
    if get_model_spec(key) is not None:
        return key, None
    listing = _get_json("https://openrouter.ai/api/v1/models")
    found = next((item for item in listing.get("data", []) if item.get("id") == raw), None)
    if found is None:
        raise HTTPException(404, f"OpenRouter has no model {raw!r} (ids look like anthropic/claude-haiku-5.5)")
    pricing = found.get("pricing") or {}
    price_in = body.price_in if body.price_in is not None else float(pricing.get("prompt") or 0) * 1e6
    price_out = body.price_out if body.price_out is not None else float(pricing.get("completion") or 0) * 1e6
    return key, {"provider": "openai_compatible", "remote": True, "api_model": raw,
                 "display_name": f"{found.get('name') or raw} (OpenRouter, added)",
                 "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
                 "price_in": round(price_in, 4), "price_out": round(price_out, 4), "energy": "not_available"}


@app.post("/api/models/custom")
def add_custom_model(body: CustomModel):
    try:
        key, entry = _custom_entry(body)
    except OSError as error:  # registry or provider not reachable
        raise HTTPException(502, f"Could not reach the model source: {error}") from error
    if entry is None:
        return {"model": public_model(get_model_spec(key)), "added": False}
    import yaml

    entries = load_custom_registry()
    entries[key] = entry
    CUSTOM_REGISTRY_PATH.write_text(yaml.safe_dump({"models": entries}, sort_keys=True))
    reload_registry()
    return {"model": public_model(get_model_spec(key)), "added": True}


@app.get("/api/catalog")
def catalog():
    return {
        "systems": [adapter.to_public_dict() for adapter in SYSTEMS.values()],
        "workloads": [workload.to_public_dict() for workload in WORKLOADS.values()],
        "models": [public_model(spec) for spec in load_registry().values()],
        "limits": {"max_run_cost_usd": settings.MAX_RUN_COST_USD, "max_repetitions": settings.MAX_REPETITIONS},
        "executor": settings.EXECUTOR,
    }


def _validate(request: RunRequest, check_credentials: bool = True):
    adapter = SYSTEMS.get(request.system)
    if adapter is None:
        raise HTTPException(400, f"Unknown system {request.system!r}")
    if adapter.status == PLANNED:
        raise HTTPException(400, f"{adapter.label} is planned but not implemented yet")
    workload = WORKLOADS.get(request.workload)
    if workload is None or request.workload not in adapter.workloads:
        raise HTTPException(400, f"{adapter.label} does not support workload {request.workload!r}")
    if request.query is not None:
        if request.query not in workload.queries:
            raise HTTPException(400, f"Unknown query {request.query!r} for {workload.dataset}")
        if not workload.query_flags:
            raise HTTPException(400, f"{workload.dataset} only runs its full query set")
    model = get_model_spec(request.model)
    if model is None:
        raise HTTPException(400, f"Model {request.model!r} is not in config/models.yaml")
    if model.provider not in adapter.providers:
        raise HTTPException(400, f"{adapter.label} does not support {model.provider} models yet")
    for operator_model in request.operator_models or []:
        if not adapter.multi_model:
            raise HTTPException(400, f"{adapter.label} has no multi-model mode")
        operator_spec = get_model_spec(operator_model)
        if operator_spec is None:
            raise HTTPException(400, f"Operator model {operator_model!r} is not in config/models.yaml")
        if check_credentials and not operator_spec.credential_available:
            raise HTTPException(400, f"{operator_spec.api_key_env} is not set on the server")
    if request.operator_models is not None and len(request.operator_models) == 1:
        raise HTTPException(400, "Multi-model mode needs at least two operator models")
    if not local_model_installed(model):
        raise HTTPException(400, f"{model.display_name} is not installed on this server")
    if check_credentials and not model.credential_available:
        raise HTTPException(400, f"{model.api_key_env} is not set on the server")
    if request.repetitions > settings.MAX_REPETITIONS:
        raise HTTPException(400, f"At most {settings.MAX_REPETITIONS} repetitions per job")
    return adapter, workload, model


@app.post("/api/runs")
async def create_run(request: RunRequest):
    adapter, workload, model = _validate(request, check_credentials=not request.dry_run)
    queries = [request.query] if request.query else workload.queries
    estimate = estimate_cost(store.records("archive"), workload.dataset, queries, model, request.repetitions)
    total = estimate["total_usd"]
    paid = model.price_in > 0 or model.price_out > 0
    if request.dry_run:
        return {"estimate": estimate}
    if paid and total is None and not request.accept_unknown_cost:
        raise HTTPException(409, {"message": "No cost estimate available; resend with accept_unknown_cost", "estimate": estimate})
    if total is not None and total > settings.MAX_RUN_COST_USD:
        raise HTTPException(
            402, {"message": f"Estimated ${total:.2f} exceeds the ${settings.MAX_RUN_COST_USD:.2f} limit", "estimate": estimate}
        )
    job_id = new_job_id()
    spec = RunSpec(
        job_id=job_id,
        system=adapter.key,
        workload=workload.key,
        model=model.key,
        query=request.query,
        repetitions=request.repetitions,
        cache_policy=request.cache_policy,
        operator_models=",".join(request.operator_models) if request.operator_models else None,
    )
    store.create_job(job_id, spec_to_dict(spec), total)
    await manager.submit(job_id)
    return {"job": store.job(job_id), "estimate": estimate}


@app.get("/api/runs")
def list_runs(limit: int = 50):
    return {"jobs": store.jobs(limit)}


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    job = store.job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return {"job": job, "records": store.records("live", job_id)}


@app.post("/api/runs/{job_id}/cancel")
async def cancel_run(job_id: str):
    if not await manager.cancel(job_id):
        raise HTTPException(409, "Job is not running")
    return {"job": store.job(job_id)}


@app.get("/api/runs/{job_id}/events")
async def run_events(job_id: str):
    if store.job(job_id) is None:
        raise HTTPException(404, "Unknown job")

    async def stream():
        async for event in manager.events(job_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/runs/{job_id}/live")
def run_live(job_id: str):
    """Live energy/calls of a measured SemBench job (Challenge mode); final record once finished."""
    job = store.job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    progress = live_progress(manager.job_dir(job_id) / "out")
    if job["status"] == "queued" or "waiting for resources" in (job.get("step") or ""):
        progress["phase"] = "waiting for a GPU"
    if job["status"] == "failed":
        progress.update(phase="failed", done=True, error=job.get("error"))
    if job["status"] == "cancelled":
        progress.update(phase="stopped", done=True)
    if job["status"] == "succeeded":
        records = store.records("live", job_id=job_id)
        progress.update(phase="finished", done=True, record=records[0] if records else None)
    elif job["status"] not in ("failed", "cancelled") and progress.get("done"):
        progress["done"] = False  # energy joined, but the server has not stored the record yet
    return {"job": job, "progress": progress}


@app.get("/api/runs/{job_id}/trace")
def run_trace(job_id: str):
    """Condensed power timeline and call activity of a finished measured run (run inspector)."""
    if store.job(job_id) is None:
        raise HTTPException(404, "Unknown job")
    windows = summarize_run_dir(manager.job_dir(job_id) / "out")
    if not windows:
        raise HTTPException(404, "No energy trace for this run")
    return {"windows": windows}


CHALLENGE_QUERIES = {"Q1", "Q2", "Q4", "Q5"}
CHALLENGE_QUALITY_TARGET = 0.8


class ChallengeEntry(BaseModel):
    name: str = Field(min_length=1, max_length=24)
    query: str
    system: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=80)
    energy_wh: float = Field(ge=0)
    quality: float = Field(ge=0, le=1)
    mode: Literal["live", "recorded"]
    job_id: Optional[str] = None


@app.get("/api/challenge/board")
def challenge_board(query: str):
    if query not in CHALLENGE_QUERIES:
        raise HTTPException(400, f"Unknown challenge query {query!r}")
    return {"entries": store.challenge_board(query, CHALLENGE_QUALITY_TARGET)}


@app.post("/api/challenge/board")
def add_challenge_entry(entry: ChallengeEntry):
    if entry.query not in CHALLENGE_QUERIES:
        raise HTTPException(400, f"Unknown challenge query {entry.query!r}")
    if entry.quality < CHALLENGE_QUALITY_TARGET:
        raise HTTPException(400, "Only correct answers are ranked")
    if entry.mode == "live":
        job = store.job(entry.job_id or "")
        if job is None or job["status"] != "succeeded":
            raise HTTPException(400, "Live entries need a finished run")
    return {"entry": store.add_challenge_entry({**entry.model_dump(), "name": entry.name.strip()})}


@app.post("/api/challenge/reset")
def reset_challenge():
    return {"deleted": store.clear_challenge()}


@app.get("/api/records")
def records(source: Literal["all", "archive", "live"] = "all"):
    return {"records": store.records(source)}
