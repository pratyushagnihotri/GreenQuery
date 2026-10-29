# Online API benchmarks

This branch runs each demo with one OpenAI model used for both ReasonDB planning and execution. The first Movie collection uses `gpt-4o-mini` and `gpt-4.1`, with Q1–Q4 and Q4's comparable Taken 3 ratio ground truth.

Online energy and CO₂ are EcoLogits estimates from OpenAI API calls, not NVML hardware measurements.  ReasonDB's application cache is disabled for every run; OpenAI provider-side caching cannot be reset or observed.

Run one validation repetition:

```bash
PYTHON_BIN=.venv/bin/python \
bash evaluation/scripts/run_movie_online_repeated.sh gpt-4o-mini \
  evaluation/demo_suites/online_api/movie/raw 1
```

After both models pass, collect ten repetitions per model and build the compact file:

```bash
PYTHON_BIN=.venv/bin/python \
bash evaluation/scripts/run_movie_online_repeated.sh gpt-4o-mini \
  evaluation/demo_suites/online_api/movie/raw 10
PYTHON_BIN=.venv/bin/python \
bash evaluation/scripts/run_movie_online_repeated.sh gpt-4.1 \
  evaluation/demo_suites/online_api/movie/raw 10
.venv/bin/python evaluation/scripts/build_online_movie_query_metrics.py \
  evaluation/demo_suites/online_api/movie/raw \
  evaluation/demo_suites/online_api/movie/data/movie_query_metrics.json
```

`data/movie_query_metrics.json` is the analysis input.  Raw measurements, sidecars, prediction files, and logs remain local provenance and are ignored by Git.  Never commit `OPENAI_API_KEY` or `.env`.

RotoWire uses the same API, EcoLogits, cache, and phase-attribution policy for its two text queries. Run only a pre-flight until you explicitly approve the full collection:

```bash
PYTHON_BIN=.venv/bin/python \
bash evaluation/scripts/run_rotowire_online_repeated.sh gpt-4o-mini \
  evaluation/demo_suites/online_api/rotowire/validation 1
```

Once approved, the compact output is built with:

```bash
.venv/bin/python evaluation/scripts/build_online_rotowire_query_metrics.py \
  evaluation/demo_suites/online_api/rotowire/raw \
  evaluation/demo_suites/online_api/rotowire/data/rotowire_query_metrics.json
```

Medical adds text and image queries Q1/Q3/Q4/Q9. Prepare the image archive before any paid API pre-flight:

```bash
.venv/bin/python evaluation/scripts/prepare_medical_dataset.py
PYTHON_BIN=.venv/bin/python \
bash evaluation/scripts/run_online_repeated.sh medical gpt-4o-mini \
  evaluation/demo_suites/online_api/medical/validation 1
PYTHON_BIN=.venv/bin/python \
bash evaluation/scripts/run_online_repeated.sh medical gpt-4.1 \
  evaluation/demo_suites/online_api/medical/validation 1
```

Once approved, build the compact output with:

```bash
.venv/bin/python evaluation/scripts/build_online_medical_query_metrics.py \
  evaluation/demo_suites/online_api/medical/raw \
  evaluation/demo_suites/online_api/medical/data/medical_query_metrics.json
```
