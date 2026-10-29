# Demo Status Matrix

This file tracks the current reproducible demo baseline for the Green Query lab work.

## Current Baseline

The initial baseline demos are:

| Demo | Modality | Reason for inclusion |
| --- | --- | --- |
| `demos/movie.py` | Text | Small text workload for query planning and dataframe-style query execution. |
| `demos/artwork.py` | Image / metadata | Small artwork workload for image-related data loading and multimodal setup. |

Other demos are useful for later coverage, but they are not the first baseline for measurement.

## Verified Status

| Demo | Branch | Machine / OS | Status | Notes |
| --- | --- | --- | --- | --- |
| `demos/movie.py` | `demo-fixes-windows-obaid` (`f0ba9ad`) | Mac, Apple Silicon | Working | Completed Q1, Q2, same-sentiment join, and Q3 count with API model. |
| `demos/artwork.py` | `demo-fixes-windows-obaid` (`f0ba9ad`) | Mac, Apple Silicon | Working after cache rebuild | Initially failed because the cached DuckDB DB missed the `artworks` table. Clearing the stale Artwork `.db` cache fixed it. |

## Smoke Test Result Details

### Movie

Command context:

```bash
PYTHONPATH=/path/to/greenQueryLab \
python demos/movie.py
```

Observed result:

```text
Benchmark positive_reviews: completed
Benchmark positive_reviews_for_taken_3: completed
Benchmark reviews_with_same_sentiment: completed
Benchmark positive_reviews_for_taken_3_count: completed
```

### Artwork

Initial failure:

```text
duckdb.duckdb.CatalogException: Catalog Error: Table with name artworks does not exist!
LINE 1: CREATE VIEW artworks AS SELECT * FROM artworks.artworks;
```

Resolution:

```text
Clear/rebuild the stale Artwork DuckDB cache DB under artworks_sampled.
```

Observed result after cache rebuild:

```text
Benchmark madonna_and_child_count_by_century:
painting_count = 1
```

