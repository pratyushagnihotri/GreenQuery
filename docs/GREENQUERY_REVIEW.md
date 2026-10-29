# GreenQuery review guide

This branch is a **reproducible review snapshot**, not a live dashboard service. It combines the project's ReasonDB/Stretto fork, local DGX evaluation path, online API evaluation path, and a Next.js workbench reading eight committed compact JSON files. Inherited Stretto planning, optimization, and semantic execution are not claimed as new GreenQuery inventions. The older Streamlit frontend (`feature/frontend`) and the separate Movie operator-telemetry branch remain historical references, not additional applications merged here.

## What came together

| Source | Head at integration | Included |
| --- | --- | --- |
| `feature/local-demo-suites` | `0a2415d` | Base: local-model runners, NVML measurement, validators, local plans |
| `feature/online-api-demo-suites` | `bfa46e5` | Merged history: API runners, response usage/cost, EcoLogits estimates, validators |
| `frontend_v2` | `dc1db6c` | Selected runtime files imported under `dashboard/`; unrelated Git history not merged |

The backend merge preserves local planning/physical-plan semantics and API phase-level usage/impact fields. This branch does **not** automatically ingest a new run, prove the local and API paths can share one live process, or demonstrate direct Caesura/Lotus compatibility.

## Follow one Movie query

```text
query/model/data/cache choices
  evaluation/scripts/run_movie_measurement.py
    -> MeasurementConfig + MeasurementWrapper (evaluation/measurement/)
    -> RaccoonDB.add_table(CSV) (reasondb/interface/connect.py)
    -> reviews.filter(...).project(...).limit(...).execute(...)
       (reasondb/interface/df.py)
    -> inherited Stretto logical planning / physical selection / execution
    -> local model or OpenAI backend call
    -> local inference + NVML collector OR API response usage + EcoLogits
    -> tracker.set_result(...) -> QueryTracker.get_result()
    -> MeasurementWrapper.__exit__ writes raw measurement / sidecars
    -> workload validator and compact-summary builder
    -> manual copy of approved compact JSON to dashboard/src/data/
    -> dashboard/src/data/benchmarks.ts -> dashboard/src/app/page.tsx
```

The runner selects a dataset, records its hash, sets the model and cache policy, and creates the ReasonDB configuration. `df.py` gives the result table its logical plan and tuned physical pipelines before executor cleanup; the wrapper records them with telemetry and the query result. A successful `.execute()` is **not** itself a validated answer. Workload validators compare the result against the available ground truth, and the compact builder creates the archived view. The dashboard reads only those static compact files on build/start; no backend endpoint or live result watcher is wired to it.

## Where to inspect or extend

| Change or question | Primary file or directory |
| --- | --- |
| Choose a Movie query, input CSV, model, cache policy, or measurement mode | `evaluation/scripts/run_movie_measurement.py` |
| Run local matched Movie/RotoWire/Medical/Animals experiments | `evaluation/demo_suites/local_matched/` and `evaluation/scripts/` (see its README and Slurm scripts) |
| Run online Movie/RotoWire/Medical/Artwork experiments | `evaluation/demo_suites/online_api/` and `evaluation/scripts/` (see its README) |
| Capture query lifecycle, results, phase evidence, plan, and output | `evaluation/measurement/wrapper.py`, `evaluation/measurement/config.py`, `evaluation/measurement/artifacts.py` |
| Attribute OpenAI response usage and EcoLogits estimates | `evaluation/measurement/api_usage.py`, `reasondb/backends/` |
| Attribute local inference/NVML GPU energy | `evaluation/measurement/local_inference.py`, `evaluation/measurement/nvml.py` |
| Configure ReasonDB model/backend and run a plan | `reasondb/interface/config.py`, `reasondb/interface/df.py`, `reasondb/optimizer/`, `reasondb/executor.py` |
| Validate answers and build compact files | `evaluation/scripts/validate_*`, `evaluation/scripts/build_*` |
| Change archived workbench metrics, quality rules, or data source | `dashboard/src/data/benchmarks.ts`, `dashboard/src/data/*.json` |
| Change workbench views, plans, or quality display | `dashboard/src/app/page.tsx`, `dashboard/src/components/` |

To add a query, define its runner expression, input identities, validation contract, and compact-output mapping; merely adding a UI label cannot make its result scientifically valid. To add a model, configure its planning/execution backend and telemetry source, then validate the same query under an explicit cache policy. To add a metric, record its source and unit in the measurement layer, carry it through the builder, and display it without mixing measured local GPU energy with estimated API impact.

An external system could be displayed at **black-box** level if an adapter emits run identity, configuration, output, validation, timing, and measurement-source metadata in the compact schema. Call-level, phase-level, or plan/operator views require the corresponding hooks or exported traces from that system. No adapter for Caesura or Lotus is implemented or tested.

## Safe setup and checks

For a clean checkout, use Python 3.12 and the inherited [root setup](../README.md) for live ReasonDB execution (submodules, model servers and the relevant Python dependencies). Local model tests/runs need DGX GPU/model access and NVML visibility; online runs need an API credential and can incur charges. Neither was exercised by the integration checks. Do not put credentials in this branch.

The archived workbench needs Node 24 or newer but no DGX/API access:

```sh
cd dashboard
npm ci
npm run verify-data
npm run lint
npm run typecheck
npx next build --webpack
npm run dev
```

`next build --webpack` is used here because Turbopack's CSS helper could not bind a localhost port in the integration sandbox. It does not need Google Fonts: the workbench uses system font fallbacks. Inspect a local and an API row in Query Lab; Movie has the structured plan examples. The [snapshot provenance note](../dashboard/DATA_PROVENANCE.md) gives the eight exact hashes and quality boundaries.

To refresh the workbench, run and validate a workload in its own environment, regenerate its compact JSON with the corresponding builder, inspect the schema/row counts/status, then deliberately replace the appropriate `dashboard/src/data/*.json` file and rerun `npm run verify-data`. This is a manual, reviewed copy; the dashboard has no automatic ingestion.
