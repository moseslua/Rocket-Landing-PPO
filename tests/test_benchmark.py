"""Benchmark receipt checks; no performance threshold or simulator tuning."""

import json
from pathlib import Path

import pytest

from rocket_landing.benchmark import BenchmarkConfig, main, run_benchmark
from rocket_landing.cli import main as cli_main


def test_benchmark_records_resolved_config_and_provenance(tmp_path):
    config = BenchmarkConfig(environments=2, steps=2, repeats=1, seed=9)
    report = run_benchmark(config)
    assert report["config"]["seed"] == 9
    assert report["metadata"]["simulator_versions"]["sim_t"]
    assert report["metadata"]["dependencies"]["scipy"]
    assert len(report["metadata"]["lock_sha256"]) == 64
    assert report["metadata"]["git"]["dirty"] in (True, False, None)
    assert report["sim_t"]["policy_transitions_per_second"] > 0
    assert report["sim_t"]["warmup_compile_seconds"] > 0
    assert len(report["reference_comparison"]) == 4
    for row in report["reference_comparison"]:
        assert row["method"] in ("RK45", "DOP853")
        assert row["nfev"] > 0
        assert row["position_error_m"] < 1e-7
        assert row["velocity_error_m_per_s"] < 1e-7
    path = tmp_path / "receipt.json"
    assert (
        main(["--environments", "1", "--steps", "1", "--repeats", "1", "--output", str(path)]) == 0
    )
    assert json.loads(path.read_text())["kind"] == "developer-numerical-benchmark"
    with pytest.raises(FileExistsError):
        main(["--environments", "1", "--steps", "1", "--repeats", "1", "--output", str(path)])
    assert not Path(str(path) + ".tmp").exists()


def test_public_benchmark_help_lists_options(capsys):
    with pytest.raises(SystemExit) as error:
        cli_main(["benchmark", "--help"])
    assert error.value.code == 0
    assert "--environments" in capsys.readouterr().out


def test_benchmark_rejects_invalid_dimensions():
    for kwargs in ({"steps": 0}, {"environments": 0}, {"repeats": -1}, {"seed": -1}):
        with pytest.raises(ValueError):
            BenchmarkConfig(**kwargs)
