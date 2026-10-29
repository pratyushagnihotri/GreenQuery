# GreenQuery evaluation — analysis (V3–V5)

Source data: `reports/sembench_measured.csv` (318 runs, 306 valid; SemBench Movie sf=1000, Q1/Q2/Q4/Q5,
1 repetition), `reports/multi_model_choices.json`. Updated 2026-10-22.

## Correct answers per model and system (single-model runs)

Score ≥ 0.8 out of 4 queries (F1 for Q1/Q2/Q5, 1 − relative error for Q4). E = engine errors
(the system failed before answering), T = 1 h timeouts.

| Model | Stretto | Abacus | LOTUS | Palimpzest | ThalamusDB | CAESURA |
|---|---|---|---|---|---|---|
| Qwen 2.5 3B | 2 E1 | 2 E1 | 1 T1 | 1 T1 | 2 | 1 |
| Qwen 2.5 7B | 2 | 2 | 2 T1 | 2 | 1 | 0 |
| Qwen 2.5 14B | 2 | 2 | 2 T1 | 3 | 2 | 0 |
| Qwen 2.5 32B | 2 | 2 | 2 T1 | 3 | 3 | 0 |
| Llama 3.2 1B | 0 E4 | 0 E4 | 1 | 1 | 0 | 0 |
| Llama 3.2 3B | 1 E3 | 1 E3 | 1 T1 | 2 | 1 | 0 |
| Llama 3.1 8B | 1 E2 | 1 E1 | 2 T1 | 2 | 1 | 1 |
| Gemma 3 4B | 2 | 2 | 1 T1 | 2 | 1 | 0 |
| Gemma 3 12B | 2 | 2 | 3 T1 | 0 T1 | 1 | 1 |
| Gemma 3 27B | 2 | 2 | 2 T1 | 3 | 3 | 1 |
| gpt-4o-mini | 2 | 2 | 2 | 2 | 2 | 1 |
| gpt-5-mini | 3 | 2 | 2 | 2 | 3 | 0 |
| gpt-4.1 | 2 | 2 | 2 | 2 | 2 | 1 |
| Claude Haiku 5.5 | 2 | 2 | 2 | 2 | 1 | 1 |
| Claude Sonnet 5.5 | 2 | 2 | 2 | 2 | 2 | 0 |
| Claude Opus 5.5 | 1 | 2 | 2 (Q5 not run) | 2 | 0 | 1 |

Observations
- **Engine errors occur only in Stretto/Abacus** (LLM planner): KeyError (12) and ReasoningDeadEnd (7),
  concentrated in the Llama family (1B: all queries; 3B: 3; 8B: 1–2) and Qwen 3B (Q1). The other
  systems never fail before answering; they return wrong answers instead.
- **Stretto's Q5 (semantic self-join) never returns a pair** on any model (cap of 2/4 for Stretto/Abacus).
- **LOTUS Q5 times out (1 h) on every local model**; it finishes only on API models (gpt-4o-mini: 18 min).
- **CAESURA** answers correctly at most once per model; its LLM writes the whole plan (wrong table / SQL).
- **ThalamusDB fails with Claude Opus 5.5 (Q1/Q2 F1 0)**: ThalamusDB asks for "1"/"0" with a 1-token answer
  limit (`max_tokens: 1`, plus an OpenAI-only logit bias). Opus starts its answer with markdown (`**`) or
  returns empty content, so no answer parses as "1"; ThalamusDB keeps sampling (869 calls vs ~20) and ends
  empty. Haiku/Sonnet answer a bare digit. Diagnosed with `--log-content` (job 49914, not in the table).
- **Bigger is better only within a family and only up to a point**: Palimpzest/ThalamusDB improve with
  Qwen 14B/32B and Gemma 27B; Stretto/Abacus plateau at 2/4 (Q5 join limits them).

## Multi-model optimizer comparison (V3; Stretto vs Abacus, Q1/Q2/Q4)

Call sequences (S = smaller operator model, L = larger; first L = planning call):

| Operators (planner) | Target | Stretto | Abacus |
|---|---|---|---|
| Qwen 3B + 32B (32B) | 0.8 | tries 3B on a sample, runs all on 32B | same as Stretto |
| Qwen 3B + 7B (7B) | 0.8 | tries 3B, runs all on 7B | same |
| Gemma 4B + 27B (27B) | 0.8 | tries 4B, runs all on 27B | **runs ~90 % on 4B**: Q1 F1 1.0 at 8.5 vs 18.8 Wh; Q2 F1 0.6 (< target) |
| Llama 3.2 3B + 3.1 8B (8B) | 0.8 | Q2 fails in planning | Q2, Q4 fail in planning |
| Qwen 3B + 32B (32B) | **0.6** | unchanged (runs on 32B) | **Q1 on 3B (88 %)**: 4.8 vs 9.6 Wh, F1 1.0 |

- Stretto's optimizer never switched to the cheaper model in our runs; Abacus did in 2 of 5 setups,
  halving GPU energy on Q1 at unchanged F1 — and once undershot its quality target (Gemma Q2).
- Both optimizers pay a visible "probing" cost (60–156 calls on the small model) even when they
  then discard it.

## Measurement checks (V4, from all runs)

- **GPU idle baseline** (models loaded, 60–120 s before each run): mean 60.5 W, sd 1.8 W, n = 36 runs,
  range 58.4–67.5 W. Higher values (63–67.5 W) occur in a few jobs, plausibly a different physical GPU;
  each run is corrected with its own baseline.
- **CPU**: the node draws 195–255 W package power (shared with other users); our job's share of busy CPU
  time while idle is 0.2–1.5 %, hence apportioning by CPU time rather than attributing node power.

## Token-based estimates vs measurement (V5, EcoLogits 0.11.1)

`evaluation/scripts/ecologits_for_runs.py` → `reports/ecologits_estimates.csv`. EcoLogits has no profile
for Qwen 2.5, Gemma 3, or Llama 3.2, so for local models we apply EcoLogits' own energy model
(`compute_llm_impacts`) with each model's parameter count, per call (completion tokens + latency), PUE 1.

| Comparison (240 local query windows) | median ratio estimate / measured | p10 – p90 |
|---|---|---|
| vs total GPU energy | 0.051 | 0.016 – 0.21 |
| vs GPU energy above idle | 0.105 | 0.036 – 0.42 |

Per system (above idle): ThalamusDB 0.05, Stretto/Abacus 0.10, CAESURA 0.11, LOTUS 0.24, Palimpzest 0.27.

- The estimate is ~10× below the measured above-idle energy and the gap differs ~5× **between systems**, so
  it cannot be corrected by one factor. Inputs are 88 % of tokens, and EcoLogits uses output tokens only.
- Ranking: for a given model and query, the estimate orders systems like the measurement in 88 % of pairs
  (521/594) — useful for coarse ranking, not for absolute energy.
- Caveat (fairness to EcoLogits): it models efficient data-centre serving; our setup is 4-bit models at
  batch 4 on one A100. Part of the gap is the serving regime, not an error of the method.
- API runs (EcoLogits provider profiles): gpt-4o-mini 28.8 Wh over 20 queries (LOTUS Q5 18.9 Wh);
  gpt-5-mini 157 Wh over 24 queries (LOTUS Q5 114 Wh); gpt-4.1 2,275 Wh over 24 queries (LOTUS Q5 2,073 Wh,
  EcoLogits assumes a much larger model). Measured local LOTUS Q5 runs: 89–341 Wh.

## Q5: who stops at the LIMIT? (V5; 10 local + 2 API models, 65,536 candidate pairs)

| System | LLM calls, median (range) | Energy Wh, median (range) | answered (F1 > 0) | timeouts |
|---|---|---|---|---|
| ThalamusDB | 4 (2–135) | 0.6 (0.3–6) | 12/12 | 0 |
| Palimpzest | 100 (always) | 5.0 (2.5–15) | 12/12 | 0 |
| CAESURA | 17 (3–39) | 8.0 (3.5–50) | 2/12 | 0 |
| Abacus (Stretto) | 1,794 (1–7,925) | 27.6 (0.5–121) | 0/12 | 0 |
| Stretto | 2,754 (1–8,108) | 45.0 (0.5–122) | 0/12 | 0 |
| LOTUS | 15,398 (2,313–33,433) | 300 (89–341) | 3/12 | 9 (all local models) |

- ThalamusDB and Palimpzest stop once ten pairs are found (ThalamusDB often after 2–4 calls); both always
  answer, at 0.3–15 Wh. LOTUS evaluates pairs without exploiting the limit: 9 of 10 local runs time out
  after 1 h (89–341 Wh); it finishes only on the API models (F1 0.33, ~18–21 min) and on the smallest,
  fastest local model, Llama 3.2 1B (20,152 calls in 29 min, F1 0.53).
- Energy for the same query spans three orders of magnitude (0.3 Wh to 341 Wh), driven by plan shape
  (early stopping), not by the model.

## Measurement validation (V4, job 49887: one GPU, phases fast/slow/fast/slow)

Same queries (ThalamusDB, LOTUS, Stretto × Q1/Q2/Q4, Qwen 2.5 7B) in alternating phases on one GPU
(`evaluation/scripts/run_measurement_validation.sbatch`, `validate_measurement.py`). Idle 61.7–62.9 W.

- **Trace overhead (0.1 s vs 5 s sampling, plus nvidia-smi at 100 ms in fast phases):** mean difference
  +3.1 % (pair 1) and 0.0 % (pair 2), both signs, within run-to-run noise — no systematic overhead.
  Run-to-run noise between identical slow phases: ~1–2 % for queries > 1 min (LOTUS Q1 10.68 vs 10.44 Wh),
  up to ~10 % for queries of a few seconds to half a minute.
- **NVML energy counter vs nvidia-smi power.draw (trapezoid, 100 ms):** within ±1 % for queries > 30 s;
  −6 % to +4 % for 6–14 s queries (few samples). The counter is the more precise of the two.
- **GPU-to-GPU variation dominates:** the earlier two-job comparison (49885 vs 49886, different physical
  A100s, idle 61.8 vs 59.5 W) differed by +8 to +20 % for identical queries. Absolute energy therefore
  carries roughly ±10–20 % hardware variation across GPUs; comparisons are most reliable within one job/GPU.

## Repetitions for the headline results (3 runs; jobs 49915, 49916, 49917, 49918)

Qwen 2.5 7B, all six systems, Q1 and Q5; multi-model Gemma 4B + 27B (Q1). Runs landed on different
physical A100s, so the spread includes GPU-to-GPU variation. Energy = GPU incl. idle + CPU share.

| System | Q1 Wh, median [min–max] | Q1 quality | Q5 Wh, median [min–max] | Q5 quality |
|---|---|---|---|---|
| ThalamusDB | 0.27 [0.24–0.28] | 1.0 ×3 | 5.6 [4.0–56] | 0.6, 0.6, 0.5 |
| Palimpzest | 3.0 [2.7–103] (1 run T/O) | 1.0, 1.0, 0.52 | 4.1 [3.8–4.2] | 0.6, 0.52, 0.31 |
| LOTUS | 16 [15–17] | 1.0 ×3 | 281 [269–298], T/O ×3 | – |
| Stretto | 6.9 [6.5–8.0] | 1.0 ×3 | 48 [44–50] | 0.0 ×3 |
| Abacus | 6.0 [5.8–7.5] | 1.0 ×3 | 39 [38–44] | 0.0 ×3 |
| CAESURA | 6.7 [5.4–10] | 0.0 ×3 | 9.8 [7.3–12] | 0.0 ×3 |

- **Headline claims hold:** for most cells the spread is 5–14 % (coefficient of variation), while systems
  differ by 10–1,000×. Q1: ThalamusDB vs LOTUS ≈ 60× in all three runs.
- **Abacus' model switch is stable:** Gemma 4B + 27B, Q1, 3 runs: Abacus 88 % of calls on 4B, GPU
  8.5/9.2/9.1 Wh, F1 1.0; Stretto 12 %, 18.8/20.2/19.9 Wh, F1 1.0.
- **Palimpzest is non-deterministic on Q1:** in 1 of 3 runs it evaluated all 1,000 reviews instead of
  stopping at the limit (no `LimitScanOp 5/5` in the log), hit the 1 h timeout (103 Wh). A single run would
  have hidden this.
- **ThalamusDB Q5 runaway generation:** ThalamusDB's join prompt has no output limit (`stop: ["."]` only,
  no `max_tokens`). In 2 of 4 runs Qwen generated until the 32k context filled (Ollama "context shift"),
  never returning; the proxy cut these calls at 600 s (1 and 4 calls), and the query still answered
  (F1 0.6) at 56 Wh / 213 Wh instead of ~5 Wh. API providers cap output by default, so this only appears
  with local models. Job 49915's run (4/114 calls cut) is marked invalid by the >1 % rule and replaced by
  49918 (1/136); both are kept on disk.
- The measurement flagged both anomalies automatically (timeouts, failed calls), which is the point of
  recording call-level evidence.

## API models: verbosity and cost

Average tokens per successful call over all runs of the model (same engines, same prompts):

| Model | calls | input / call | output / call |
|---|---|---|---|
| gpt-4o-mini | 70,389 | 130 | 3.1 |
| gpt-5-mini (reasoning_effort minimal) | 32,916 | 182 | 13.3 |
| gpt-4.1 | 34,500 | 190 | 36.3 |
| Claude Opus 5.5 | 9,328 | 187 | 66.7 |
| Claude Sonnet 5.5 | 36,819 | 274 | 145.8 |
| Claude Haiku 5.5 | 61,913 | 287 | 261.4 |

- Engines ask for one-word answers (yes/no, 1/0); Claude models answer with sentences: Haiku writes ~84×
  more output per call than gpt-4o-mini. Output tokens cost 5× input, so verbosity, not model size,
  drives cost: Haiku LOTUS Q5 cost $4.87 (8.1M output tokens), Sonnet LOTUS Q5 $66.45.
- Claude also counts ~1.5× more input tokens for the same prompts (tokenizer).

Cost per model (`api_cost_usd` per query in `reports/sembench_measured.csv`; `cost_source` says how it was
determined: `provider_billed` = charged amount returned by OpenRouter per call; `list_price_no_cache` =
tokens × list price, an upper bound, for runs before cache tracking):

| Model | Cost | Source |
|---|---|---|
| gpt-4o-mini | $1.15 (latest runs; ≈ $1.7 incl. reruns) | list price |
| gpt-5-mini | $2.37 | list price |
| gpt-4.1 | $23.14 (LOTUS Q5 $20.00) | list price |
| Claude Haiku 5.5 | $5.27 | provider-billed (LOTUS Q5 rerun) + list price |
| Claude Sonnet 5.5 | $73.80 (LOTUS Q5 $66.45) | provider-billed |
| Claude Opus 5.5 | $19.40 (LOTUS Q5 not run: est. ≈ $170) | provider-billed |

OpenRouter total incl. smoke tests and diagnostics: $103.49 (of a $200 budget).

EcoLogits for Claude (no 5.5 profiles; Claude 4.5 stand-ins from `config/models.yaml`,
`reports/ecologits_claude.csv`): Haiku 877 Wh, Sonnet 9,521 Wh (LOTUS Q5 alone 8,737 Wh), Opus 3,189 Wh over
their queries. These stand-in estimates are driven by assumed model sizes and should not be compared with
measured local energy.

## Not run
- Repetitions beyond the headline results (user decision: 1 run per configuration elsewhere).
