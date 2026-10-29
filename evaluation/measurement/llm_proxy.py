"""Logging, model-aliasing proxy for OpenAI-compatible LLM traffic.

Every engine under test (Stretto, LOTUS, Palimpzest, ThalamusDB, CAESURA) is pointed at
this proxy as if it were the OpenAI API (OPENAI_BASE_URL / OPENAI_API_BASE). The proxy

* forwards each request to an upstream OpenAI-compatible server (Ollama on the GPU node,
  or api.openai.com),
* optionally rewrites the model name (``--alias gpt-4o-mini=qwen2.5:7b-instruct``) so engines
  with a fixed model list run the local model under test unchanged,
* appends one JSON line per call: start/end time, path, requested and served model,
  prompt/completion tokens, status,
* optionally enforces a spending cap (``--budget-usd`` with ``--price-in/--price-out`` in USD
  per million tokens): once the logged spend reaches the cap, further calls get HTTP 402
  without reaching the upstream. Calls already in flight may overshoot it slightly,
* optionally adapts requests for reasoning models (``--reasoning-effort``): drops sampling
  parameters they reject and renames max_tokens; each call logs what was changed,
* optionally (``--no-prefill``, Claude 5.5) turns a final assistant prefill into a "Continue." turn.

The call log is joined with the GPU energy trace afterwards (per-call / per-query energy),
so all engines are measured the same way without modifying them.

Run:  python -m evaluation.measurement.llm_proxy --port 9100 \
          --upstream http://127.0.0.1:11434/v1 --log calls.jsonl \
          --alias gpt-4o-mini=qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

HOP_BY_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "accept-encoding",
              "content-encoding"}  # upstream is asked for identity encoding; bodies are relayed decoded


class CallLog:
    def __init__(self, path: Path, price_in: float = 0.0, price_out: float = 0.0,
                 price_cached_in: Optional[float] = None):
        self.path = path
        self.lock = threading.Lock()
        self.price_in, self.price_out = price_in, price_out
        self.price_cached_in = price_in if price_cached_in is None else price_cached_in
        self.spent_usd = 0.0
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict) -> None:
        """Log one call with its cost: the provider's charged amount when the response carries one
        (OpenRouter usage.cost), else tokens x price with cached input tokens at the cached price."""
        prompt = record.get("prompt_tokens") or 0
        cached = min(record.get("cached_tokens") or 0, prompt)
        computed = ((prompt - cached) * self.price_in + cached * self.price_cached_in
                    + (record.get("completion_tokens") or 0) * self.price_out) / 1e6
        billed = record.get("billed_usd")
        cost, source = (float(billed), "provider_billed") if billed is not None else (computed, "tokens_x_price")
        with self.lock, self.path.open("a") as handle:
            self.spent_usd += cost
            record = {**record, "cost_usd": cost, "cost_computed_usd": computed, "cost_source": source,
                      "spent_usd": self.spent_usd}
            handle.write(json.dumps(record) + "\n")


def usage_from(body: Dict) -> Dict[str, Optional[float]]:
    """Token usage from an OpenAI-style (usage) or Ollama-native (eval counts) response, plus cached
    input tokens (OpenAI prompt_tokens_details) and the charged amount when present (OpenRouter usage.cost)."""
    usage = body.get("usage") or {}
    prompt = usage.get("prompt_tokens", body.get("prompt_eval_count"))
    completion = usage.get("completion_tokens", body.get("eval_count"))
    details = usage.get("prompt_tokens_details") or {}
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "cached_tokens": details.get("cached_tokens"), "billed_usd": usage.get("cost")}


REASONING_UNSUPPORTED = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs",
                         "stop")
# Reasoning tokens count against max_completion_tokens: engines that allow a handful of output tokens
# (yes/no answers) would get no answer at all, so small limits are raised to this floor.
REASONING_MIN_COMPLETION_TOKENS = 1024


def adapt_for_reasoning(payload: Dict, effort: str) -> list:
    """Make a chat request written for gpt-4o-mini acceptable to a reasoning model (gpt-5 family).

    Drops parameters these models reject (sampling, logprobs, stop), renames max_tokens, raises small
    completion limits to REASONING_MIN_COMPLETION_TOKENS, and sets reasoning_effort.
    Returns the list of changes so each call records how its request was adapted.
    """
    changes = [key for key in REASONING_UNSUPPORTED if key in payload]
    for key in changes:
        payload.pop(key)
    if "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        changes.append("max_tokens->max_completion_tokens")
    limit = payload.get("max_completion_tokens")
    if isinstance(limit, int) and limit < REASONING_MIN_COMPLETION_TOKENS:
        payload["max_completion_tokens"] = REASONING_MIN_COMPLETION_TOKENS
        changes.append(f"max_completion_tokens {limit}->{REASONING_MIN_COMPLETION_TOKENS}")
    if "messages" in payload and "reasoning_effort" not in payload:
        payload["reasoning_effort"] = effort
        changes.append(f"reasoning_effort={effort}")
    return changes


def adapt_no_prefill(payload: Dict) -> list:
    """Models without assistant prefill (Claude 5.5): a conversation must end with a user message.

    When an engine ends with a partial assistant message for the model to continue, append a short user
    turn asking to continue, so the model completes the same text. Returns the change for the call log.
    """
    messages = payload.get("messages")
    if isinstance(messages, list) and messages and messages[-1].get("role") == "assistant":
        messages.append({"role": "user", "content": "Continue."})
        return ["assistant_prefill->continue"]
    return []


def build_app(upstream: str, log: CallLog, aliases: Dict[str, str], api_key: Optional[str],
              budget_usd: Optional[float] = None, reasoning_effort: Optional[str] = None,
              no_prefill: bool = False, request_cost: bool = False, log_content: bool = False) -> Starlette:
    # Engines send full paths (/v1/chat/completions); forward them relative to the server root
    # so both "http://host:11434" and "http://host:11434/v1" work as --upstream.
    root = upstream.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    client = httpx.AsyncClient(base_url=root, timeout=httpx.Timeout(600.0))

    async def forward(request: Request) -> Response:
        path = request.path_params["path"]
        raw = await request.body()
        payload = None
        requested = served = None
        adapted: list = []
        if raw and request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict) and "model" in payload:
            requested = payload["model"]
            served = aliases.get(requested, requested)
            payload["model"] = served
            if reasoning_effort:
                adapted = adapt_for_reasoning(payload, reasoning_effort)
            if no_prefill:
                adapted += adapt_no_prefill(payload)
            if payload.get("stream"):
                # Ask for usage in the final chunk so streamed calls are counted too.
                payload.setdefault("stream_options", {"include_usage": True})
            if request_cost:  # OpenRouter: include the charged amount (usage.cost) in every response
                payload["usage"] = {"include": True}
            raw = json.dumps(payload).encode()
        if budget_usd is not None and log.spent_usd >= budget_usd:
            log.write({"started_at": time.time(), "finished_at": time.time(), "path": "/" + path,
                       "requested_model": requested, "served_model": served, "status": 402,
                       "stream": False, "prompt_tokens": 0, "completion_tokens": 0, "budget_exceeded": True})
            message = f"GreenQuery proxy budget of ${budget_usd:.2f} reached; call not forwarded"
            return Response(json.dumps({"error": {"message": message, "type": "budget_exceeded"}}),
                            status_code=402, media_type="application/json")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["accept-encoding"] = "identity"  # else httpx asks for gzip and the relayed body/header mismatch
        if api_key:  # real key lives only in the proxy, not in the engines
            headers["authorization"] = f"Bearer {api_key}"
        started = time.time()
        upstream_request = client.build_request(
            request.method, "/" + path, content=raw, headers=headers, params=request.query_params
        )
        try:
            response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as error:  # upstream unreachable / dropped: log it as a failed call
            log.write({"started_at": started, "finished_at": time.time(), "path": "/" + path,
                       "requested_model": requested, "served_model": served, "status": 502, "stream": False,
                       "prompt_tokens": None, "completion_tokens": None, "error": f"{type(error).__name__}: {error}"})
            return Response(json.dumps({"error": {"message": f"upstream error: {type(error).__name__}",
                                                  "type": "upstream_error"}}),
                            status_code=502, media_type="application/json")
        record = {
            "started_at": started, "path": "/" + path, "requested_model": requested,
            "served_model": served, "status": response.status_code,
            "stream": bool(isinstance(payload, dict) and payload.get("stream")),
        }
        if adapted:
            record["adapted"] = adapted
        content_type = response.headers.get("content-type", "")

        if record["stream"] or "text/event-stream" in content_type:
            async def relay():
                usage = {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None, "billed_usd": None}
                buffer = b""
                try:
                    async for chunk in response.aiter_raw():
                        buffer += chunk
                        yield chunk
                finally:
                    await response.aclose()
                    for line in buffer.decode(errors="replace").splitlines():
                        if line.startswith("data: ") and '"usage"' in line:
                            try:
                                found = usage_from(json.loads(line[6:]))
                                if found["prompt_tokens"] is not None:
                                    usage = found
                            except json.JSONDecodeError:
                                pass
                    log.write({**record, **usage, "finished_at": time.time()})
            out_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP}
            return StreamingResponse(relay(), status_code=response.status_code, headers=out_headers)

        body = await response.aread()
        await response.aclose()
        if response.status_code >= 400:  # keep the upstream reason for failed calls (truncated)
            record["error_body"] = body[:500].decode(errors="replace")
        elif log_content:  # debugging: the start of the model's answer
            try:
                record["answer_head"] = (json.loads(body)["choices"][0]["message"]["content"] or "")[:200]
            except (ValueError, KeyError, IndexError, TypeError):
                pass
        usage = {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None, "billed_usd": None}
        try:
            usage = usage_from(json.loads(body))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            pass
        log.write({**record, **usage, "finished_at": time.time()})
        out_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP}
        return Response(content=body, status_code=response.status_code, headers=out_headers)

    async def health(_: Request) -> Response:
        return Response(json.dumps({"status": "ok", "upstream": upstream, "aliases": aliases,
                                    "spent_usd": log.spent_usd, "budget_usd": budget_usd}),
                        media_type="application/json")

    return Starlette(routes=[
        Route("/proxy-health", health),
        Route("/{path:path}", forward, methods=["GET", "POST", "PUT", "DELETE"]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:11434/v1 or https://api.openai.com/v1")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--alias", action="append", default=[], help="requested=served model name rewrite")
    parser.add_argument("--api-key-env", default=None, help="env var with the upstream key (OpenAI upstream only)")
    parser.add_argument("--price-in", type=float, default=0.0, help="USD per million prompt tokens")
    parser.add_argument("--price-out", type=float, default=0.0, help="USD per million completion tokens")
    parser.add_argument("--budget-usd", type=float, default=None, help="refuse calls once this spend is reached")
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"],
                        help="adapt requests for reasoning models (gpt-5 family) and set this effort")
    parser.add_argument("--price-cached-in", type=float, default=None,
                        help="USD per million cached prompt tokens (default: --price-in)")
    parser.add_argument("--request-cost", action="store_true",
                        help="ask the upstream for the charged amount per call (OpenRouter usage.cost)")
    parser.add_argument("--log-content", action="store_true",
                        help="debugging: log the first 200 characters of each (non-streamed) answer")
    parser.add_argument("--no-prefill", action="store_true",
                        help="models without assistant prefill: add a 'Continue.' user turn after a final assistant message")
    args = parser.parse_args()
    aliases = dict(item.split("=", 1) for item in args.alias)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    import uvicorn

    uvicorn.run(build_app(args.upstream, CallLog(args.log, args.price_in, args.price_out, args.price_cached_in),
                          aliases, api_key, args.budget_usd, args.reasoning_effort, args.no_prefill,
                          args.request_cost, args.log_content),
                host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
