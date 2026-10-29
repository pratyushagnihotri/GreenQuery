# Setup Issues and Fixes

This file records setup findings needed to reproduce the current Movie and Artwork demo baseline.

## Common Baseline

| Item | Current value / note |
| --- | --- |
| Repository | `DataManagementLab/greenQueryLab` |
| Tested branch | `demo-fixes-windows-obaid` |
| Verified commit | `f0ba9ad` |
| Python | Python 3.10 environment |
| DuckDB | `duckdb==1.2.0` |
| Models | OpenAI API-backed configuration |
| Initial demos | `demos/movie.py`, `demos/artwork.py` |

## API Key

The demos require `OPENAI_API_KEY` to be set in the shell environment.

Do not commit API keys into the repository.

```bash
export OPENAI_API_KEY="..."
```

If the key is missing, demos fail during model backend construction with:

```text
KeyError: 'OPENAI_API_KEY'
```

## Python Path When Testing Worktrees

When testing a separate Git worktree with an existing editable install, Python may import `reasondb` from the old editable checkout instead of the worktree being tested.

For branch verification, force imports to use the target worktree:

```bash
PYTHONPATH=/path/to/greenQueryLab python demos/movie.py
```

## Artwork Cache Issue

Artwork can fail after image downloads with:

```text
duckdb.duckdb.CatalogException: Catalog Error: Table with name artworks does not exist!
LINE 1: CREATE VIEW artworks AS SELECT * FROM artworks.artworks;
```

This indicates that the `artworks_sampled` DuckDB cache DB exists but does not contain the expected `artworks` table.

Minimal fix:

```bash
find ~/.reasondb/cache/artworks_sampled -maxdepth 1 \( -name "*.db" -o -name "*.db.wal" \) -delete
```

Then rerun:

```bash
python demos/artwork.py
```

If testing with a temporary home/cache directory, apply the same deletion to that cache root.

## Network Requirements

Artwork downloads referenced images from Wikimedia during first run. Without network access, it can fail before query execution with a `requests.exceptions.ConnectionError`.

After the images are cached, reruns should be much faster and should avoid most image downloads.

## Movie Device Note

On Obaid's branch, `demos/movie.py` still contains:

```python
with RaccoonDB("movie_sampled", device="cuda:3") as rc:
```

The Mac smoke test still completed because the selected plans used traditional operators and did not hit a CUDA-only optimizer path. If GPU-specific optimizer paths are exercised on Mac, changing the device to `cpu` may still be necessary.

