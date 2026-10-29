# SemBench, vendored for GreenQuery

**Upstream:** https://github.com/SemBench/SemBench, commit `c556cae469d7fe4873ab2ab337493cdb58ab5de3`
(2026-10-12; also in `UPSTREAM_COMMIT`). **License:** MIT or Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`),
unchanged.

GreenQuery runs the real semantic query engines through SemBench's runners and evaluator. This copy lets
GreenQuery run SemBench from its own repository (`SEMBENCH_ROOT` defaults to `third_party/sembench`; set the
variable to use another checkout).

## What is included

Only what GreenQuery needs, exported with `git archive` at the pinned commit: `src/`, `config/`, `scripts/`,
`requirements/`, `requirements.txt`, `README.md`, `ENVIRONMENT_SETUP.md`, the licenses, and for the Movie
scenario `files/movie/query/`, `files/movie/raw_results/ground_truth/` and `files/movie/data/sf_2000/`.
Not included: `docs/` (156 MB), `figures/`, the other scenarios' files — fetch them from upstream at the
same commit if needed.

## GreenQuery changes (every changed line is marked `# GreenQuery`)

Also available as a patch against the pinned commit: `integrations/sembench/sembench-greenquery.patch`
(`git apply` it to a fresh SemBench checkout).

| File | Change |
|---|---|
| `src/run.py` | registers the `stretto` and `stretto_abacus` systems |
| `src/scenario/movie/runner/stretto_runner/`, `stretto_abacus_runner/` | GreenQuery's runners for Stretto and Abacus-in-Stretto (source of truth: `integrations/sembench/`) |
| `src/runner/generic_caesura_runner/caesura/main.py`, `model.py` | models not in CAESURA's tables get default limits instead of a `KeyError` |
| `src/runner/generic_caesura_runner/generic_caesura_runner.py` | constructor argument order; pass the model name through (was a silent fallback to `gpt-4-0613`); current model prices |
| `src/scenario/movie/runner/caesura_runner/caesura_runner.py` | data file names `Movies.csv` / `Reviews.csv` (current data generator) |
| `src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py` | model config lookup falls back to a file derived from the model name |
| `config/system/thalamusdb/gpt_4o_mini.json` | ThalamusDB configuration for gpt-4o-mini (same structure as upstream's other model files) |

## Runtime setup (not in git)

- **Per-system environments** `.venvs/<system>/` (lotus, palimpzest, thalamusdb, caesura, sembench; ~9 GB):
  create them with SemBench's setup (`ENVIRONMENT_SETUP.md`, `scripts/`) inside this folder, or link existing
  ones from another SemBench checkout. `stretto` and `stretto_abacus` link to
  GreenQuery's `.venv` (`integrations/sembench/install.sh` creates these links).
- **Generated data** `files/movie/data/sf_<n>/` (we use sf 1000): produced by SemBench's setup from the
  source data, or copied from an existing checkout.
- **Run output** `files/*/metrics/`, `files/*/raw_results/<system>/`: written by every run; GreenQuery's job
  scripts copy what they need into `tests/results/`. Ignored by git.
