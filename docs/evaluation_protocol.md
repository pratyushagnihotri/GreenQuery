# Evaluation Protocol

This file defines the initial measurement plan for the Green Query lab work.

## Purpose

The goal is to move from "the demos run" to reproducible measurements of trade-offs between:

| Category | Metrics |
| --- | --- |
| Correctness / quality | Output correctness, precision, recall, F1 where ground truth exists |
| Runtime | Total demo runtime, query runtime, per-operator runtime if available |
| API usage | Number of model calls, token usage, estimated monetary cost |
| Energy | Estimated energy for API calls or local execution |
| Carbon | Estimated CO2 emissions |
| Reliability | Failures, retries, tracebacks, cache issues |

## Initial Demo Scope

Only the first two baseline demos should be used for initial measurements:

| Demo | Workload type | Initial role |
| --- | --- | --- |
| `demos/movie.py` | Text | First measurement target because it is smaller and easier to interpret. |
| `demos/artwork.py` | Image / metadata | Second measurement target after Movie is stable. |

## Run Types

### Smoke Test

Purpose: verify that the demo starts, executes, and returns a result.

Record:

```text
demo name
branch
commit
machine / OS
Python version
DuckDB version
model configuration
status: pass / fail
traceback if failed
```

### Stabilization Test

Purpose: verify that the same demo works across multiple team machines.

Record:

```text
team member
machine / OS
branch
whether cache was cleared
runtime rough estimate
status
notes
```

### Measurement Run

Purpose: collect comparable runtime, cost, energy, and output-quality data.

Record:

```text
demo name
query name
model name
start time
end time
total runtime
number of API calls if available
token usage if available
estimated API cost if available
energy estimate
CO2 estimate
result summary
errors / warnings
```

## Energy Tool Plan

| Execution mode | Tool | Notes |
| --- | --- | --- |
| API / remote LLMs | EcoLogits | Use for OpenAI/API model calls where local hardware energy is not the full picture. |
| Local/offline models | CodeCarbon | Use for CPU/GPU/RAM energy estimates during local execution. |
| DGX / server GPU runs | NVML or DCGM | Use later for GPU-level measurements on shared servers. |

## First Measurement Target

Start with:

```text
demos/movie.py + API model + EcoLogits
```

Reason:

```text
Movie is the confirmed text baseline and is less complex than Artwork because it does not need first-run image downloads.
```

After Movie measurement is stable, repeat the same structure for:

```text
demos/artwork.py + API model + EcoLogits
```

For Artwork, separate first-run download/setup time from warm-cache query runtime.

