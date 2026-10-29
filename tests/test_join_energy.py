import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "evaluation" / "scripts"))

from join_energy import join, merge


def trace_rows():
    # GPU counter rises 100 J/s; node busy CPU 10 s/s, job uses 2 s/s; package 200 J/s.
    rows = [{"meta": True}]
    for t in range(0, 21):
        rows.append({"t": float(t), "gpu_j": {"0": 100.0 * t}, "cpu_pkg_j": 200.0 * t,
                     "job_cpu_s": 2.0 * t, "node_busy_cpu_s": 10.0 * t})
    return rows


def test_merge_overlapping_intervals():
    assert merge([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]


def test_join_computes_gpu_cpu_and_call_energy():
    windows = [
        {"kind": "idle", "started_at": 0.0, "finished_at": 5.0},
        {"kind": "query", "system": "lotus", "query": "Q1", "repetition": 1,
         "started_at": 10.0, "finished_at": 20.0, "exit_code": 0},
    ]
    calls = [
        {"started_at": 11.0, "finished_at": 13.0, "prompt_tokens": 100, "completion_tokens": 2, "status": 200},
        {"started_at": 12.0, "finished_at": 14.0, "prompt_tokens": 50, "completion_tokens": 1, "status": 200},
        {"started_at": 30.0, "finished_at": 31.0, "prompt_tokens": 999, "completion_tokens": 9, "status": 200},
    ]
    (record,) = join(trace_rows(), calls, windows)
    assert record["gpu_energy_j"] == pytest.approx(1000.0)        # 10 s x 100 W
    assert record["gpu_idle_w"] == pytest.approx(100.0)            # idle window at same rate
    assert record["gpu_dynamic_energy_j"] == pytest.approx(0.0)
    assert record["llm_calls"] == 2                                # call outside window excluded
    assert record["prompt_tokens"] == 150 and record["completion_tokens"] == 3
    assert record["llm_active_s"] == pytest.approx(3.0)            # union of [11,13] and [12,14]
    assert record["gpu_energy_in_llm_calls_j"] == pytest.approx(300.0)
    assert record["cpu_job_share"] == pytest.approx(0.2)
    assert record["cpu_job_energy_j"] == pytest.approx(400.0)      # 2000 J node x 0.2
