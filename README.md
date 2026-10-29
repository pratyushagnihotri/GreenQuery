<p align="center">
  <img src="docs/images/greenquery_logo.webp" alt="GreenQuery" width="640">
</p>

<p align="center"><b>Measure the quality, energy, runtime and cost of LLM-powered semantic query engines.</b></p>

GreenQuery runs semantic query engines (Stretto, Abacus, LOTUS, Palimpzest, ThalamusDB, CAESURA) on the same
SemBench queries, the same data, the same LLM and the same GPU, without modifying any engine. A logging proxy
records every LLM call, and an energy tracer samples the GPU (NVML) and CPU (RAPL) every 100 ms, so every
query gets **measured** energy, quality, runtime, tokens and API cost. A Next.js dashboard lets you explore
the results, start new runs on a GPU cluster, and play the **GreenQuery Challenge**: answer a query correctly
with the least energy and beat the built-in optimizers.

> In 441 runs over 6 engines and 16 LLMs, correct answers to the same query differ by up to 513× in energy.
> Switching the engine at a fixed model changes energy by a median of 13×, while growing the model from 3B to 32B
> changes it by 2.7×. Energy cannot be read off LLM calls, tokens or dollars. Details: [`docs/EXPERIMENTAL_SETUP.md`](docs/EXPERIMENTAL_SETUP.md).

---

## Contents

- [How it works](#how-it-works)
- [What the dashboard offers](#what-the-dashboard-offers)
- [Prerequisites](#prerequisites)
- [Quick start: dashboard only (no GPU)](#quick-start-dashboard-only-no-gpu)
- [Full setup: server + dashboard + live runs](#full-setup-server--dashboard--live-runs)
- [Running the query engines and optimizers](#running-the-query-engines-and-optimizers)
- [From raw runs to the dashboard](#from-raw-runs-to-the-dashboard)
- [Configuration reference](#configuration-reference)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
 Dashboard (Next.js, :3100) ──HTTP/SSE──► GreenQuery server (FastAPI, :8765 on the DGX login node)
                                             │  sbatch, one job per run
                                             ▼
                              SLURM job on 1 exclusive A100 GPU
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │  Engine (unmodified)  ──"gpt-4o-mini"──►  Logging proxy  ──►  Ollama (model     │
   │  LOTUS / Palimpzest /                     calls.jsonl         under test)       │
   │  ThalamusDB / CAESURA /                   (tokens, timing,    or OpenAI /       │
   │  Stretto / Abacus                          cost, status)       OpenRouter API   │
   │                                                                                 │
   │  Energy tracer (100 ms): NVML GPU energy counter, RAPL CPU package, CPU times   │
   │  Query windows: windows.jsonl         SemBench evaluator: quality per query     │
   └───────────────────────────────────────────────────────────────────────────────┘
                                             │  join_energy.py
                                             ▼
                  per-query records: quality · energy · runtime · calls · tokens · cost
```

- **Every engine is configured to call OpenAI "gpt-4o-mini".** The proxy rewrites the model name to the model
  under test, so engine configurations are identical across models, and a call that bypasses the proxy fails
  instead of reaching a paid API.
- **Measured energy** (local models): `E = E_GPU (incl. idle) + E_CPU-package × job's share of CPU time`.
- **API models** (OpenAI, Claude via OpenRouter): inference runs at the provider, so energy is **estimated** with
  EcoLogits and always labelled as such. It is never mixed with measured energy. API cost is the billed cost.
- Quality is computed by SemBench's own evaluator. An answer is *correct* if quality ≥ 0.8.

---

## What the dashboard offers

The dashboard works in two modes. **Recorded**: it ships a snapshot of all measured runs and works offline,
with no server. **Live**: connected to the GreenQuery server, it starts real runs on the GPU cluster and streams
their progress.

### Overview: the headline in one screen
Key findings (energy spread for the same answer, optimizers beaten, correct answers, API cost) and a
quality-vs-energy plot of every engine and query for one model. Below it: archive coverage, measurement
provenance (measured vs. estimated) and per-dataset summaries.

![Overview](docs/images/screen_overview.png)

### Challenge: play the query optimizer
Pick a query, an engine and a model, and try to answer correctly with the least energy. Your run races the
built-in optimizers (Stretto, Abacus) on a live power trace with one tick per LLM call. Correct answers enter a
shared leaderboard.
- **Recorded mode** replays measured runs (accelerated) and works offline.
- **Live on the GPU** runs the query on the DGX with measured energy.
- **Bring your own model**: add any Ollama model (e.g. `phi3:mini`). The next run downloads it.
- **Play on your phone**: the QR code opens the Challenge on phones in the same network (or on the public
  deployment, see `NEXT_PUBLIC_PLAY_URL`).

![Challenge, run in progress](docs/images/screen_challenge.png)

### Cross-System: same query, same model, same GPU
An energy heatmap of engines × queries for one model. Each cell shows energy and quality (correct, partly
correct, wrong, timeout), and the cheapest correct answer per query is outlined. It also has a quality-vs-energy
scatter, run details (GPU energy above idle, CPU share, tokens, raw run id) and a per-query summary ("energy
differs 59× between systems", energy spent on wrong answers).

![Cross-System](docs/images/screen_cross_system.png)

### Run Benchmark: start runs from the browser
Choose system, workload, query, model, repetitions, cache mode and, for Stretto/Abacus, single- or multi-model
operators. You get a cost estimate before starting (runs above the cost limit are refused), then live job cards
with energy, LLM calls, runtime and a power sparkline. **Inspect** opens the full trace of a run.

![Run Benchmark](docs/images/screen_run_benchmark.png)

### Query Lab: where the energy of one run goes
A single recorded run, broken down into planner (LLM), execution (per operator) and everything outside both
(startup, data loading, idle). It also shows the recorded physical plan as a graph, plus runtime, energy, CO₂,
tokens, requests and quality.

![Query Lab](docs/images/screen_query_lab.png)

### Model Compare: one query, several models
Query-level medians per model (quality, runtime, energy, CO₂), with chart views for quality, latency vs.
energy, accuracy vs. energy, CO₂ and the Pareto frontier.

![Model Compare](docs/images/screen_model_compare.png)

### Compare Queries: workloads side by side
Several queries under one model: question and complexity, the planner vs. execution energy share, the
run-to-run spread (median, IQR, min/max) and a summary table.

![Compare Queries](docs/images/screen_compare_queries.png)

### Dataset Compare: benchmark coverage
Dataset-level medians (Animals, Artwork, Medical, Movie, RotoWire), their quality metrics and phase coverage,
plus a data audit of what the archive can and cannot support.

![Dataset Compare](docs/images/screen_dataset_compare.png)

### Measurement Records: the raw evidence
Every individual recorded run behind the aggregates (query, model, run, evaluation, energy, phase coverage), not
medians.

![Measurement Records](docs/images/screen_measurement_records.png)

---

## Prerequisites

| For | You need |
|---|---|
| Dashboard only (recorded mode) | **Node.js ≥ 24** and npm |
| GreenQuery server | **Python 3.12**, [uv](https://docs.astral.sh/uv/) (or pip), the repo's `.venv` with `reasondb` installed |
| Live runs with measured energy | A **SLURM cluster with NVIDIA GPUs** (we use a DGX A100, one 40 GB A100 per job), NVML access (`nvidia-smi`), RAPL readable through unprivileged `perf` (no root needed), [Ollama](https://ollama.com) binary and a model store |
| SemBench engines | The vendored SemBench in `third_party/sembench` with one virtual environment per engine in `third_party/sembench/.venvs/<system>` (see [`third_party/sembench/GREENQUERY.md`](third_party/sembench/GREENQUERY.md)) |
| API models (optional) | `OPENAI_API_KEY` and/or `OPENROUTER_API_KEY` in a repo-root `.env` (never committed) |
| Paper figures (optional) | `uv run --with matplotlib`, `--with ecologits==0.11.1` for API energy estimates |

Without SLURM the server runs jobs as local subprocesses (`GREENQUERY_EXECUTOR=local`). In that mode only
Stretto runs. The SemBench engines and measured energy need the cluster.

---

## Quick start: dashboard only (no GPU)

The dashboard ships the measured snapshot (`dashboard/src/data/sembench_measured.json`), so every page works
without a server.

```bash
cd dashboard
npm install
npm run dev -- -p 3100
```

Open <http://localhost:3100>. Live features show "backend offline", and the Challenge runs in recorded mode
with a leaderboard stored in your browser.

Production build:

```bash
cd dashboard
npm run check          # lint + typecheck + build
npm run start -- -p 3100
```

---

## Full setup: server + dashboard + live runs

### 1. Clone and create the Python environment (cluster login node)

```bash
git clone https://github.com/DataManagementLab/greenQueryLab.git
cd greenQueryLab
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e . -r greenquery_server/requirements.txt
```

The full Stretto setup (PyTorch, kvpress, local model servers for the original Stretto experiments) is
described in [`docs/STRETTO_README.md`](docs/STRETTO_README.md). It is not needed for the SemBench runs, which
serve models through Ollama.

### 2. Secrets (`.env` in the repo root, `chmod 600`)

```bash
OPENAI_API_KEY=...        # only for OpenAI models and planner calls
OPENROUTER_API_KEY=...    # only for Claude via OpenRouter
```

The server and the job scripts read `.env`. Keys are passed only to the logging proxy, while the engines get
a dummy key.

### 3. SemBench and the engines

```bash
integrations/sembench/install.sh          # registers the Stretto/Abacus runners, links their venv
```

Create the per-engine environments `third_party/sembench/.venvs/{lotus,palimpzest,thalamusdb,caesura,sembench}`
with SemBench's own setup (`third_party/sembench/ENVIRONMENT_SETUP.md`), and generate the Movie data at scale
factor 1000 (`files/movie/data/sf_1000/`).

### 4. Ollama models

```bash
export OLLAMA_MODELS=~/ollama-models       # model store used by the jobs
ollama pull qwen2.5:7b-instruct            # and any other model you want to run
ollama pull nomic-embed-text               # embeddings for engines that need them
```

Each job starts its own Ollama server on its GPU. The store only has to contain the models.

### 5. Start the GreenQuery server (cluster login node)

```bash
scripts/start_server_dgx.sh               # SLURM mode, listens on 127.0.0.1:8765
curl -s localhost:8765/api/health          # {"status":"ok"}
```

Logs go to `logs/greenquery_server.log` and the PID to `logs/greenquery_server.pid`. Running the script again
restarts the server.

Without a cluster (laptop, local executor):

```bash
.venv/bin/uvicorn greenquery_server.app:app --port 8000
```

### 6. Connect the dashboard (your laptop)

```bash
# tunnel: laptop port 8010 -> server port 8765 on the cluster
ssh -N -L 8010:localhost:8765 <user>@<cluster-login-node>

# dashboard against that API, reachable from phones in the same network
cd dashboard
NEXT_PUBLIC_GREENQUERY_API=http://localhost:8010 npx next dev --webpack -H 0.0.0.0 -p 3100
```

Open <http://localhost:3100>. The footer shows live records, and **Run Benchmark** and the Challenge's
**Live on the GPU** mode are enabled. Phones that open the dashboard through the QR code reach the API through
the dashboard's `/gq-api` proxy, so the tunnel only needs to exist on the laptop.

---

## Running the query engines and optimizers

You can start runs **from the dashboard** (Run Benchmark, Challenge) or **directly with SLURM**. Both produce the
same output folder `tests/results/<job>-<id>/`.

### Engines

| Key | Engine | How it answers the query |
|---|---|---|
| `stretto` | Stretto | LLM planner + gradient-descent plan optimizer over physical operators |
| `stretto_abacus` | Abacus (in Stretto) | Abacus' Pareto-cascade optimizer, same operators and executor as Stretto |
| `lotus` | LOTUS | Semantic operators, hand-written SemBench program |
| `palimpzest` | Palimpzest | Declarative, cost-based plans, hand-written SemBench program |
| `thalamusdb` | ThalamusDB | Approximate processing with early stop, SemBench SQL |
| `caesura` | CAESURA | LLM writes the plan (SQL + Python) from natural language |

Queries (SemBench Movie, scale factor 1000): **Q1** five positive reviews, **Q2** five positive reviews of one
movie, **Q4** share of positive reviews of one movie, **Q5** ten review pairs with the same sentiment
(semantic self-join).

### Local model, measured energy

```bash
# defaults: qwen2.5:7b-instruct, LOTUS/Palimpzest/ThalamusDB/CAESURA, Q1 Q2 Q4 Q5, 1 repetition
sbatch evaluation/scripts/run_sembench_measured.sbatch

# choose model, engines, queries and repetitions
sbatch --export=ALL,LOCAL_MODEL=gemma3:12b,SYSTEMS="lotus palimpzest",QUERIES="1 5",REPETITIONS=3 \
  evaluation/scripts/run_sembench_measured.sbatch

# the optimizers: Stretto and Abacus
sbatch --export=ALL,LOCAL_MODEL=qwen2.5:7b-instruct,SYSTEMS="stretto stretto_abacus" \
  evaluation/scripts/run_sembench_measured.sbatch
```

### Multi-model optimizers (small + large model per operator)

Stretto and Abacus may implement each operator with any of the listed models. `LOCAL_MODEL` is the planner.
The model list contains commas, so set it in the environment rather than in `--export`:

```bash
GREENQUERY_OPERATOR_MODELS="ollama/qwen2.5:3b-instruct,ollama/qwen2.5:32b-instruct" \
  sbatch --export=ALL,LOCAL_MODEL=qwen2.5:32b-instruct,SYSTEMS="stretto stretto_abacus",OLLAMA_CONTEXT_LENGTH=8192 \
  evaluation/scripts/run_sembench_measured.sbatch
```

`GREENQUERY_QUALITY_TARGET` (default 0.8) sets the optimizers' quality target. `OLLAMA_CONTEXT_LENGTH=8192` lets
a 32B model and a second model share one 40 GB GPU.

### API models (OpenAI, Claude via OpenRouter)

```bash
# gpt-4o-mini with a hard spending cap (the proxy answers HTTP 402 once reached)
sbatch --export=ALL,BUDGET_USD=5,SYSTEMS="lotus thalamusdb",QUERIES="1 2" evaluation/scripts/run_sembench_openai.sbatch

# another OpenAI model; gpt-5 family needs a reasoning effort
sbatch --export=ALL,SERVED_MODEL=gpt-5-mini,REASONING_EFFORT=minimal,PRICE_IN=0.25,PRICE_OUT=2.0 \
  evaluation/scripts/run_sembench_openai.sbatch

# Claude through OpenRouter (set PRICE_IN/PRICE_OUT from config/models.yaml)
sbatch --export=ALL,PROVIDER=openrouter,SERVED_MODEL=anthropic/claude-haiku-5.5,PRICE_IN=<in>,PRICE_OUT=<out> \
  evaluation/scripts/run_sembench_openai.sbatch
```

Prices (USD per million tokens) are in [`config/models.yaml`](config/models.yaml). API runs need no GPU.

### Useful job variables

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_MODEL` | `qwen2.5:7b-instruct` | Ollama model under test |
| `SYSTEMS` | `lotus palimpzest thalamusdb caesura` | engines to run |
| `QUERIES` | `1 2 4 5` | SemBench Movie query ids |
| `REPETITIONS` / `REP_OFFSET` | `1` / `0` | repetitions, and numbering offset for extra runs |
| `QUERY_TIMEOUT` | `3600` | seconds per (engine, query). A timeout scores 0 |
| `IDLE_SECONDS` | `60` | idle window before the first query (idle baseline) |
| `TRACE_INTERVAL` | `0.1` | energy sampling period in seconds |
| `SCALE_FACTOR` | `1000` | SemBench Movie scale factor |

### Original Stretto benchmarks

The Stretto paper's optimizer experiments (`scripts/run_benchmark.py`, executors `optim_global`, `abacus`,
`lotus`, ...) are documented in [`docs/STRETTO_README.md`](docs/STRETTO_README.md).

---

## From raw runs to the dashboard

Each job writes `tests/results/<job>-<id>/` with `calls.jsonl` (every LLM call), `trace.jsonl` (energy samples),
`windows.jsonl` (query windows), `run_config.txt`, SemBench metrics, and `energy_records.json`. The job script
creates `energy_records.json` itself with `join_energy.py`.

```bash
# 1. (only if a job was interrupted) join energy, calls and windows per query
.venv/bin/python evaluation/scripts/join_energy.py tests/results/sembench-measured-<id>

# 2. combine all runs into one table: engine x model x query x repetition
.venv/bin/python evaluation/scripts/combine_sembench_measured.py tests/results/sembench-* \
  --out reports/sembench_measured

# 3. export it as the dashboard snapshot
.venv/bin/python evaluation/scripts/export_dashboard_snapshot.py reports/sembench_measured.csv \
  --out dashboard/src/data/sembench_measured.json

# 4. (optional) EcoLogits energy estimates for the API runs
uv run --no-project --with ecologits==0.11.1 python evaluation/scripts/ecologits_api_sembench.py \
  tests/results --out reports/ecologits_api_sembench.csv

# 5. (optional) paper figures
uv run --no-project --with matplotlib python paper/edbt2027-demo/figures/make_fig_pareto.py
uv run --no-project --with matplotlib python paper/edbt2027-demo/figures/make_fig_motivation_q1q5.py
```

---

## Configuration reference

**Server** (environment variables or `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `GREENQUERY_EXECUTOR` | `local` | `local` = subprocesses, `slurm` = one `sbatch` job per run |
| `GREENQUERY_DATA_DIR` | `server_data/` | job folders and the SQLite store (leaderboard, records) |
| `GREENQUERY_PYTHON` | current interpreter | Python used for runner subprocesses |
| `GREENQUERY_MAX_RUN_COST_USD` | `5` | refuse runs whose estimated API cost is higher |
| `GREENQUERY_MAX_REPETITIONS` | `10` | per run |
| `GREENQUERY_MAX_CONCURRENT_JOBS` | `1` | local executor |
| `GREENQUERY_MEASURED_IDLE_SECONDS` | `20` | idle baseline for runs started from the dashboard |
| `GREENQUERY_MEASURED_QUERY_TIMEOUT` | `900` | per-query timeout for runs started from the dashboard |
| `GREENQUERY_CUSTOM_MODEL_MAX_GB` | `20` | largest Ollama model a visitor may add |
| `GREENQUERY_CORS_ORIGINS` | `http://localhost:3000,http://localhost:3100` | allowed dashboard origins |
| `SEMBENCH_ROOT` | `third_party/sembench` | SemBench checkout |
| `OLLAMA_MODELS` | `~/ollama-models` | Ollama model store |

**Dashboard** (set when starting `next dev` / `next build`):

| Variable | Meaning |
|---|---|
| `NEXT_PUBLIC_GREENQUERY_API` | server URL, e.g. `http://localhost:8010` through the SSH tunnel |
| `NEXT_PUBLIC_PLAY_URL` | address the Challenge QR code points to (e.g. a public deployment). Default: this page's network address |

**Models**: [`config/models.yaml`](config/models.yaml) lists every selectable model with provider, prices, energy
source (measured, EcoLogits, or EcoLogits proxy) and GPU needs.

---

## Repository layout

```
dashboard/                Next.js dashboard (pages in src/components, snapshot in src/data)
greenquery_server/        FastAPI server: catalog, jobs (local/SLURM), live progress, leaderboard
  adapters/               how each engine family is launched and its records collected
evaluation/scripts/       SLURM job scripts, logging proxy, energy tracer, join/combine/export, EcoLogits
integrations/sembench/    Stretto and Abacus runners for SemBench, SemBench patch, installer
third_party/sembench/     vendored SemBench (pinned commit c556cae, changes marked "# GreenQuery")
config/models.yaml        model registry (providers, prices, energy source)
reasondb/                 Stretto / ReasonDB engine
reports/                  combined results (CSV/Markdown), EcoLogits estimates
paper/edbt2027-demo/      EDBT 2027 demo paper, figures, tables, screenshots
docs/                     setup, evaluation plan, experimental setup, known issues
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard shows "backend offline" | Is the SSH tunnel running? `curl localhost:8010/api/health`. Was the dashboard started with `NEXT_PUBLIC_GREENQUERY_API`? |
| Phone can't open the QR link | Phone and laptop must be on the same Wi-Fi. Allow incoming connections for `node` in the macOS firewall prompt. Start `next dev` with `-H 0.0.0.0`, and restart it after changing networks. Conference Wi-Fi often blocks this: use a public deployment and `NEXT_PUBLIC_PLAY_URL`. |
| Job refuses to start: "GPU already has a compute process" | The job only runs on an exclusive GPU. Wait, or let SLURM pick another GPU. |
| Large models (27B/32B) run out of memory | Set `OLLAMA_CONTEXT_LENGTH=8192`. |
| SemBench engines are listed as "planned" | They need `GREENQUERY_EXECUTOR=slurm` and the per-engine venvs in `third_party/sembench/.venvs/`. |
| Known engine behaviour | Stretto's Q5 join returns no pairs ([`docs/STRETTO_Q5_JOIN_ISSUE.md`](docs/STRETTO_Q5_JOIN_ISSUE.md)). Engines are run unmodified. See also [`docs/known_issues.md`](docs/known_issues.md). |

---

## Credits

GreenQuery builds on [Stretto](https://arxiv.org/abs/2602.04430) (Gabriele Sanmartino, Matthias Urban, Paolo Papotti, Carsten Binnig) and uses workload (query and data) from
[SemBench](https://github.com/SemBench/SemBench). It runs LOTUS, Palimpzest/Abacus, ThalamusDB and CAESURA
unmodified through SemBench's runners. Energy estimates for API models use [EcoLogits](https://ecologits.ai).
