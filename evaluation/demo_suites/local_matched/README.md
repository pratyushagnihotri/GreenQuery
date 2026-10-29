# Local matched demo suites

This package compares two fully local models on DGX: `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`. In every run, the same model plans and executes the query. No OpenAI API or remote planner is used.

All runs use cold caches and record NVML GPU energy, derived CO2, runtime, tokens, requests, and available ground-truth quality.

| Demo | Queries | Dataset | Quality status |
| --- | --- | --- | --- |
| Movie | Q1-Q4 | `reviews_100.csv`; Q4 uses the corrected 120-row Taken 3 ratio subset | Deterministically scored |
| RotoWire | R1-R2 | player/game reports | Current compact data has valid resource metrics; do not present its old quality as accuracy. Recollect with the included visible `game_id` + `name` scorer for quality. |
| Medical | Q1, Q3, Q4, Q9 | Medical text and image data | Ground-truth metadata is recorded; Q3/Q9 use local LLaVA. |

## Run on DGX

Set the project environment first. `PYTHON_BIN` must point to the DGX virtual environment and `GREENQUERY_REPO_ROOT` to this checkout.

```bash
export GREENQUERY_REPO_ROOT="$PWD"
export PYTHON_BIN=/path/to/.venv/bin/python

sbatch evaluation/scripts/run_movie_demo_repeated.sbatch Qwen/Qwen2.5-7B-Instruct 10 results/movie
sbatch evaluation/scripts/run_movie_demo_repeated.sbatch meta-llama/Llama-3.1-8B-Instruct 10 results/movie

sbatch evaluation/scripts/run_rotowire_demo_repeated.sbatch Qwen/Qwen2.5-7B-Instruct 10 results
sbatch evaluation/scripts/run_rotowire_demo_repeated.sbatch meta-llama/Llama-3.1-8B-Instruct 10 results

python evaluation/scripts/prepare_medical_dataset.py
sbatch evaluation/scripts/run_medical_demo_repeated.sbatch Qwen/Qwen2.5-7B-Instruct 10 results/medical
sbatch evaluation/scripts/run_medical_demo_repeated.sbatch meta-llama/Llama-3.1-8B-Instruct 10 results/medical
```

Run Qwen and Llama sequentially unless Slurm gives distinct GPUs and cache namespaces. Medical needs two GPUs per job: one text-model GPU and one local LLaVA GPU. Validate a one-repetition preflight before a 10-repetition collection:

```bash
python evaluation/scripts/validate_matched_demo.py path/to/measurement.json --model Qwen/Qwen2.5-7B-Instruct --demo movie
python evaluation/scripts/build_matched_demo_summary.py results --output results/query_metrics.json
```

## Presentation data and notebook

`data/` contains compact chart inputs only: `movie_query_metrics.json` (80 rows), `rotowire_query_metrics.json` (40 rows), and `medical_query_metrics.json` (80 rows). Raw JSON, predictions, sidecars, and Slurm logs are provenance and intentionally ignored by Git.

```bash
marimo edit evaluation/demo_suites/local_matched/analysis/local_matched_analytics.py
PYTHONPATH=evaluation/demo_suites/local_matched/analysis python -m unittest evaluation/demo_suites/local_matched/analysis/test_local_matched_analytics_data.py
```

The one notebook has a sidebar selector for Movie, RotoWire, and Medical. Its resource, paired-median, planner-versus-execution-token, and quality sections adapt to the chosen dataset.

Data dictionary: `energy_kwh` is NVML query-window GPU energy; `co2_g` is energy multiplied by the configured emissions factor; `runtime_seconds` is end-to-end query time; planner/execution token fields and request fields are local-model telemetry; `quality` is task-specific ground-truth evidence. ReasonDB `fake_cost` is not a measurement and is excluded.

Newly collected compact rows also include `phase_energy_valid`, `phase_energy_provenance`, planner/execution energy, CO2 and NVML sample counts, plus unattributed query energy and CO2. The dashboard shows this split only when all selected runs have valid `nvml_non_overlapping_local_inference_windows_v1` provenance.
