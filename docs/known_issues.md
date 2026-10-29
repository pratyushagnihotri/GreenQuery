# Known Issues

This file separates known setup/demo issues from benchmark results.

## Artwork: Missing `artworks` Table In Cache DB

Status: resolved by cache rebuild during Mac verification.

Symptom:

```text
duckdb.duckdb.CatalogException: Catalog Error: Table with name artworks does not exist!
LINE 1: CREATE VIEW artworks AS SELECT * FROM artworks.artworks;
```

Likely cause:

```text
Stale or incomplete DuckDB cache DB under artworks_sampled.
```

Minimal recovery:

```bash
find ~/.reasondb/cache/artworks_sampled -maxdepth 1 \( -name "*.db" -o -name "*.db.wal" \) -delete
```

Then rerun:

```bash
python demos/artwork.py
```

## Movie: CUDA Device Setting

`demos/movie.py` on Obaid's branch still uses:

```python
device="cuda:3"
```

The Mac smoke test completed successfully, but this remains a portability risk if optimizer paths require GPU execution. If CUDA-specific failures appear on local machines, use `cpu` for local smoke tests.

## Movie: Optimizer Assertion On Some Query Shapes

Previously observed with dataframe query shapes such as:

```text
filter -> project -> limit
```

The optimizer can treat `project` or `limit` as a dependent level and hit an assertion expecting only `level == 0`.

This should be tracked as an optimizer limitation, not as a blocker for the initial Movie/Artwork baseline if the current branch run completes.

## BioDEX: Smoke Test Only

BioDEX was previously run only as a reduced compatibility smoke test. Any reduced `SAMPLE_SIZE` setting should not be treated as a final benchmark result.

