import json

import pytest

from asymmetric_autoencoder.evaluation.reporting import (
    aggregate,
    collect,
    load_run,
    markdown_table,
    mean_std,
)


def _make_run(root, name, seed, mse, with_eval=True):
    run = root / f"{name}_seed{seed}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"best_epoch": 3, "best_val_mse": mse, "parameters": 100})
    )
    (run / "history.json").write_text(
        json.dumps([{"epoch": e, "val_mse": mse + 1 / e, "seconds": 2.0} for e in (1, 2, 3)])
    )
    if with_eval:
        metrics = {"mse": mse, "mae": 2 * mse, "psnr": 20.0, "ssim": 0.9}
        (run / "eval_val.json").write_text(json.dumps(metrics))
        (run / "eval_test.json").write_text(json.dumps(metrics))
    return run


def test_mean_std():
    assert mean_std([1.0, 3.0]) == (2.0, pytest.approx(2**0.5))
    assert mean_std([5.0]) == (5.0, 0.0)
    with pytest.raises(ValueError):
        mean_std([])


def test_collect_aggregate_and_table(tmp_path):
    for seed, mse in enumerate([0.01, 0.02, 0.03]):
        _make_run(tmp_path, "m", seed, mse)
    runs = collect(tmp_path, "m", [0, 1, 2])
    agg = aggregate(runs)
    assert agg["n_seeds"] == 3 and agg["parameters"] == 100
    assert agg["test_mse"]["mean"] == pytest.approx(0.02)
    assert agg["test_mse"]["std"] == pytest.approx(0.01)
    assert agg["seconds_per_epoch"]["mean"] == 2.0
    table = markdown_table(
        {"m": agg},
        [("test_mse", "MSE", ".3f"), ("parameters", "P", ","), ("latency_x", "L", ".1f")],
    )
    assert "0.020 ± 0.010" in table and "| 100 |" in table and "n/a" in table


def test_missing_run_fails_loudly(tmp_path):
    _make_run(tmp_path, "m", 0, 0.01)
    with pytest.raises(FileNotFoundError):
        collect(tmp_path, "m", [0, 1])
    (tmp_path / "bad_seed0").mkdir()
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "bad_seed0")
