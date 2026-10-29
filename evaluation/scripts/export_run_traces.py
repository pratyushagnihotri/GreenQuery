"""Export condensed power timelines of measured runs for the dashboard's run inspector.

  python evaluation/scripts/export_run_traces.py tests/results/sembench-measured-* \
      --out dashboard/public/inspector

Writes one JSON per query window, <run>__<system>__<query>__rep<n>.json (evaluation/measurement/
trace_summary.py), fetched by the dashboard on demand. Only local (measured) runs have energy traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.measurement.trace_summary import summarize_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("dashboard/public/inspector"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    written = total = 0
    for run in args.runs:
        if not (run / "trace.jsonl").exists() or not (run / "windows.jsonl").exists():
            continue
        for summary in summarize_run_dir(run):
            name = f"{run.name}__{summary['system']}__{summary['query']}__rep{summary['repetition']}.json"
            payload = json.dumps(summary, separators=(",", ":"))
            (args.out / name).write_text(payload)
            written += 1
            total += len(payload)
    print(f"{written} query windows -> {args.out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
