import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "evaluation" / "scripts"))

from evaluation.measurement.energy_counters import (
    JOULES_PER_KWH,
    CounterSample,
    CounterSampler,
    RaplDomain,
    RaplReader,
    summarize_window,
)
from compare_energy_methods import COUNTER_ONLY_SOURCE, SAMPLER_SOURCE, compare


def make_sampler(samples, rapl_domains=()):
    sampler = CounterSampler(interval_seconds=0.1, gpu_indices=["0"])
    sampler.samples = [CounterSample(t, gpu, rapl) for t, gpu, rapl in samples]
    sampler.rapl.domains = list(rapl_domains)
    return sampler


PACKAGE = RaplDomain("package-0", "package", Path("/nonexistent"), 0)
DRAM = RaplDomain("package-0/dram", "dram", Path("/nonexistent"), 0)


def test_window_energy_interpolates_cumulative_counters():
    sampler = make_sampler([(0.0, {"0": 0.0}, {}), (1.0, {"0": 100.0}, {}), (2.0, {"0": 300.0}, {})])
    joules = sampler.joules_between(0.5, 1.5)
    assert joules["gpu"]["0"] == pytest.approx(150.0)


def test_window_is_clipped_to_sampled_range():
    sampler = make_sampler([(10.0, {"0": 0.0}, {}), (12.0, {"0": 200.0}, {})])
    assert sampler.joules_between(0.0, 100.0)["gpu"]["0"] == pytest.approx(200.0)
    assert sampler.joules_between(20.0, 30.0) == {"gpu": {}, "rapl": {}}


def test_rapl_reader_handles_counter_wraparound(tmp_path):
    zone = tmp_path / "intel-rapl:0"
    zone.mkdir()
    (zone / "name").write_text("package-0\n")
    (zone / "max_energy_range_uj").write_text("1000\n")
    (zone / "energy_uj").write_text("900\n")
    reader = RaplReader(root=tmp_path)
    assert reader.available
    first = reader.read()["package-0"]
    (zone / "energy_uj").write_text("100\n")  # wrapped past 1000
    second = reader.read()["package-0"]
    assert second - first == pytest.approx(200 / 1e6)


def test_unreadable_rapl_is_reported_not_zero(tmp_path):
    reader = RaplReader(root=tmp_path / "missing")
    assert not reader.available
    assert "rapl_unavailable" in reader.error


def test_summary_reports_boundary_idle_and_phases(tmp_path, monkeypatch):
    sampler = make_sampler(
        [
            (0.0, {"0": 0.0}, {"package-0": 0.0, "package-0/dram": 0.0}),
            (10.0, {"0": 2000.0}, {"package-0": 500.0, "package-0/dram": 100.0}),
        ],
        [PACKAGE, DRAM],
    )
    idle = tmp_path / "idle.json"
    idle.write_text(json.dumps({
        "gpu_watts_by_gpu": {"0": 50.0},
        "rapl_watts_by_kind": {"package": 30.0, "dram": 5.0},
    }))
    monkeypatch.setenv("GREENQUERY_IDLE_BASELINE", str(idle))
    summary = summarize_window(
        sampler, 0.0, 10.0,
        phase_windows={"planner": [(0.0, 2.0)], "execution": [(5.0, 9.0)]},
        co2_g_per_kwh=344.0,
    )
    assert summary["gpu_energy_kwh"] == pytest.approx(2000 / JOULES_PER_KWH)
    assert summary["cpu_package_energy_kwh"] == pytest.approx(500 / JOULES_PER_KWH)
    assert summary["dram_energy_kwh"] == pytest.approx(100 / JOULES_PER_KWH)
    assert summary["measured_energy_kwh"] == pytest.approx(2600 / JOULES_PER_KWH)
    assert summary["measured_boundary"] == "gpu+cpu_package+dram"
    # 2000 J - 50 W * 10 s
    assert summary["gpu_dynamic_energy_kwh"] == pytest.approx(1500 / JOULES_PER_KWH)
    assert summary["cpu_package_dynamic_energy_kwh"] == pytest.approx(200 / JOULES_PER_KWH)
    phases = summary["phases"]
    assert phases["planner"]["gpu_energy_kwh"] == pytest.approx(400 / JOULES_PER_KWH)
    assert phases["execution"]["gpu_energy_kwh"] == pytest.approx(800 / JOULES_PER_KWH)
    assert phases["unattributed_gpu_energy_kwh"] == pytest.approx(800 / JOULES_PER_KWH)
    assert summary["measured_co2_g"] == pytest.approx(2600 / JOULES_PER_KWH * 344)


def test_summary_without_rapl_has_no_node_total():
    sampler = make_sampler([(0.0, {"0": 0.0}, {}), (5.0, {"0": 500.0}, {})])
    summary = summarize_window(sampler, 0.0, 5.0)
    assert summary["gpu_energy_kwh"] == pytest.approx(500 / JOULES_PER_KWH)
    assert summary["cpu_package_energy_kwh"] is None
    assert summary["measured_energy_kwh"] is None


def test_compare_reports_method_difference_and_sampler_overhead():
    base = {"demo": "movie", "query": "Q1", "model": "qwen"}
    records = [
        {**base, "energy_source": SAMPLER_SOURCE, "energy_kwh": 1.1, "counter_gpu_energy_kwh": 1.0,
         "counter_cpu_package_energy_kwh": 0.3, "counter_dram_energy_kwh": 0.1, "counter_measured_energy_kwh": 1.4},
        {**base, "energy_source": COUNTER_ONLY_SOURCE, "energy_kwh": 1.0, "counter_gpu_energy_kwh": 1.0,
         "counter_cpu_package_energy_kwh": 0.2, "counter_dram_energy_kwh": 0.1, "counter_measured_energy_kwh": 1.3},
    ]
    (row,) = compare(records)
    assert row["sampler_vs_counter_gpu"] == pytest.approx(0.1)
    assert row["sampler_overhead"] == pytest.approx(1.4 / 1.3 - 1)
    assert row["runs_with_sampler"] == 1 and row["runs_counter_only"] == 1
