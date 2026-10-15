"""Aggregate per-run result files into mean/std tables across seeds."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0.0 when only one value is available)."""
    if not values:
        raise ValueError("no values")
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load everything recorded for one run directory (missing optional files are omitted)."""
    run = Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(run)}
    for key, name in (
        ("summary", "summary.json"),
        ("history", "history.json"),
        ("val", "eval_val.json"),
        ("test", "eval_test.json"),
        ("efficiency", "efficiency.json"),
    ):
        path = run / name
        if path.exists():
            out[key] = json.loads(path.read_text())
    if "summary" not in out:
        raise FileNotFoundError(f"{run} has no summary.json (training not finished?)")
    return out


def collect(results_dir: str | Path, experiment: str, seeds: list[int]) -> list[dict[str, Any]]:
    """Runs of ``experiment`` for ``seeds`` (fails loudly if any is missing)."""
    return [load_run(Path(results_dir) / f"{experiment}_seed{s}") for s in seeds]


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std over runs for the quantities reported in the core comparison."""
    agg: dict[str, Any] = {"n_seeds": len(runs)}

    def add(key: str, values: list[float]) -> None:
        m, s = mean_std(values)
        agg[key] = {"mean": m, "std": s, "values": values}

    add("best_epoch", [r["summary"]["best_epoch"] for r in runs])
    add("best_val_mse_training", [r["summary"]["best_val_mse"] for r in runs])
    add(
        "seconds_per_epoch",
        [statistics.fmean(h["seconds"] for h in r["history"]) for r in runs],
    )
    add("total_train_seconds", [sum(h["seconds"] for h in r["history"]) for r in runs])
    agg["parameters"] = runs[0]["summary"]["parameters"]
    for split in ("val", "test"):
        if all(split in r for r in runs):
            for metric in ("mse", "mae", "psnr", "ssim"):
                add(f"{split}_{metric}", [r[split][metric] for r in runs])
    if all("efficiency" in r for r in runs):
        eff0 = runs[0]["efficiency"]
        agg["flops"] = eff0["flops"]
        for key in eff0["latency"]:
            add(f"latency_{key}_ms", [r["efficiency"]["latency"][key]["median_ms"] for r in runs])
    return agg


def markdown_table(rows: dict[str, dict[str, Any]], metrics: list[tuple[str, str, str]]) -> str:
    """``metrics`` = [(aggregate key, column title, format spec)].

    Scalar keys have no std; with a single run only the value is shown (no ± term).
    """
    header = "| Model | " + " | ".join(t for _, t, _ in metrics) + " |"
    sep = "|---|" + "---|" * len(metrics)
    lines = [header, sep]
    for name, agg in rows.items():
        cells = []
        for key, _, fmt in metrics:
            v = agg.get(key)
            if v is None:
                cells.append("n/a")
            elif isinstance(v, dict):
                if len(v["values"]) < 2:  # a standard deviation from one run would be meaningless
                    cells.append(f"{v['mean']:{fmt}}")
                else:
                    cells.append(f"{v['mean']:{fmt}} ± {v['std']:{fmt}}")
            else:
                cells.append(f"{v:{fmt}}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
