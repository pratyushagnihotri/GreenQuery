"""Pre-run API cost estimates from archived token usage and registry prices."""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Optional

from reasondb.reasoning.providers import ModelSpec


def estimate_cost(
    archive: List[Dict], dataset: str, queries: List[str], spec: ModelSpec, repetitions: int
) -> Dict[str, Optional[float]]:
    """Estimate USD for a job.

    Uses the median input/output tokens each query consumed in archived online
    runs of the same model (or of any model when it has no history), priced at
    the selected model's registry rates. When no
    archived online run exists for a query the estimate is reported as unknown.
    """
    per_query: Dict[str, Optional[float]] = {}
    for query in queries:
        rows = [
            record for record in archive
            if record.get("dataset") == dataset and record.get("query") == query
            and isinstance(record.get("input_tokens"), (int, float))
        ]
        if not rows:
            per_query[query] = None
            continue
        # Token usage differs a lot between models (e.g. Medical: gpt-4o-mini uses
        # ~18x the tokens of gpt-4.1), so prefer the selected model's own history.
        same_model = [record for record in rows if record.get("model") in (spec.key, spec.api_model)]
        rows = same_model or rows
        tokens_in = median(record["input_tokens"] for record in rows)
        tokens_out = median(record.get("output_tokens") or 0 for record in rows)
        per_query[query] = (tokens_in * spec.price_in + tokens_out * spec.price_out) / 1_000_000
    known = [value for value in per_query.values() if value is not None]
    total = sum(known) * repetitions if len(known) == len(per_query) else None
    return {"total_usd": total, "per_query_usd": per_query, "basis": "median archived tokens x registry price"}
