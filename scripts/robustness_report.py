"""Tables and figures from robustness_val.json files (single noise seed per run)."""

import argparse
import json
from pathlib import Path

from asymmetric_autoencoder.utils.visualization import robustness_curves

MODELS = {
    "baseline": "baseline_cifar10",
    "baseline_matched": "baseline_matched_cifar10",
    "asymmetric": "asymmetric_cifar10",
}
KINDS = {"gaussian": "Gaussian noise sigma", "salt_pepper": "salt-and-pepper probability"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fig-dir", default="reports/figures")
    args = parser.parse_args()
    root = Path(args.results_dir)
    data = {
        n: json.loads((root / f"{e}_seed{args.seed}" / "robustness_val.json").read_text())
        for n, e in MODELS.items()
    }
    lines = []
    for kind, xlabel in KINDS.items():
        levels = sorted(data["baseline"][kind], key=float)
        lines += [
            f"\n### {kind}: PSNR dB / SSIM vs clean target (validation, N1)\n",
            "| level | corrupted input | " + " | ".join(MODELS) + " |",
            "|---|---|" + "---|" * len(MODELS),
        ]
        for lv in levels:
            inp = data["baseline"][kind][lv]
            cells = [
                "identical to clean"
                if inp["input_mse"] == 0
                else f"{inp['input_psnr']:.2f} / {inp['input_ssim']:.3f}"
            ]
            cells += [
                f"{data[m][kind][lv]['psnr']:.2f} / {data[m][kind][lv]['ssim']:.3f}" for m in MODELS
            ]
            lines.append(f"| {lv} | " + " | ".join(cells) + " |")
        xs = [float(lv) for lv in levels]
        for metric, label in (("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
            robustness_curves(
                xs,
                {m: [data[m][kind][lv][metric] for lv in levels] for m in MODELS},
                [data["baseline"][kind][lv][f"input_{metric}"] for lv in levels],
                Path(args.fig_dir) / f"robustness_{kind}_{metric}.png",
                f"{label} under {kind.replace('_', '-')} corruption (val, n=1 seed)",
                xlabel,
                label,
            )
    out = root / "robustness_val.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
