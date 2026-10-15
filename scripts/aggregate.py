"""Aggregate core-experiment runs into tables and figures (mean +/- std over seeds)."""

import argparse
import json
from pathlib import Path

import torch

from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.data.datasets import build_test_set
from asymmetric_autoencoder.evaluation.reporting import aggregate, collect, markdown_table
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.training.checkpoints import load_checkpoint
from asymmetric_autoencoder.utils.visualization import curves, reconstruction_grid, seed_bars

MODELS = {
    "baseline": "baseline_cifar10",
    "baseline_matched": "baseline_matched_cifar10",
    "asymmetric": "asymmetric_cifar10",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    results = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    runs = {name: collect(results, exp, args.seeds) for name, exp in MODELS.items()}
    agg = {name: aggregate(r) for name, r in runs.items()}
    (results / "core_comparison.json").write_text(json.dumps(agg, indent=2))

    s = args.split
    n = len(args.seeds)
    note = f"n={n} seed{'s' if n > 1 else ''}"
    columns = [
        ("parameters", "Params", ","),
        ("flops", "FLOPs/img", ","),
        (f"{s}_mse", f"{s} MSE", ".5f"),
        (f"{s}_mae", f"{s} MAE", ".4f"),
        (f"{s}_psnr", f"{s} PSNR (dB)", ".2f"),
        (f"{s}_ssim", f"{s} SSIM", ".4f"),
        ("best_epoch", "best epoch", ".1f"),
        ("seconds_per_epoch", "s/epoch", ".1f"),
        ("latency_cpu_batch1_ms", "CPU lat. b=1 (ms)", ".2f"),
        ("latency_mps_batch128_ms", "MPS lat. b=128 (ms)", ".2f"),
    ]
    table = markdown_table(agg, columns)
    (results / f"core_comparison_{s}.md").write_text(table + "\n")
    print(table)

    fig_dir.mkdir(parents=True, exist_ok=True)
    curves(
        {n: [[h["val_mse"] for h in r["history"]] for r in rs] for n, rs in runs.items()},
        fig_dir / "val_mse_curves.png",
        f"Validation MSE during training ({note})",
        "validation MSE (final resolution)",
    )
    for metric, label in (("mse", "MSE"), ("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
        key = f"{s}_{metric}"
        seed_bars(
            {n: agg[n][key]["values"] for n in agg},
            fig_dir / f"{s}_{metric}.png",
            f"{s.capitalize()} {label} ({note}; dots = individual seeds)",
            label,
        )

    if not args.no_grid and s == "test":
        cfg0 = load_config(results / "baseline_cifar10_seed0" / "config.yaml")
        test = build_test_set(cfg0.data)
        x = torch.stack([test[i][0] for i in range(8)])
        recs = {}
        for name, exp in MODELS.items():
            cfg = load_config(results / f"{exp}_seed{args.seeds[0]}" / "config.yaml")
            model = build_model(cfg.model, 3, cfg.data.image_size)
            load_checkpoint(
                results / f"{exp}_seed{args.seeds[0]}" / "checkpoints" / "best.pt", model
            )
            model.eval()
            with torch.no_grad():
                recs[name] = model(x)
        reconstruction_grid(
            x,
            recs,
            fig_dir / "reconstructions_test.png",
            f"CIFAR-10 test images 0-7, seed {args.seeds[0]} best checkpoints",
        )


if __name__ == "__main__":
    main()
