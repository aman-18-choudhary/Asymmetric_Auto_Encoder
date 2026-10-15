"""Noise-robustness evaluation (protocol N1: clean-trained model, corrupted input, clean target).

Runs on the VALIDATION split only; the official test split is reserved and never used here.
Writes <run_dir>/robustness_val.json for each run.
"""

import argparse
import json
from pathlib import Path

from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.data.datasets import build_train_val, make_loader
from asymmetric_autoencoder.evaluation.robustness import evaluate_corruption
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.training.checkpoints import load_checkpoint
from asymmetric_autoencoder.utils.logging import get_logger, setup_logging
from asymmetric_autoencoder.utils.reproducibility import resolve_device, seed_everything

GAUSSIAN = [0.0, 0.05, 0.1, 0.2, 0.3]
SALT_PEPPER = [0.02, 0.05, 0.1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories")
    parser.add_argument("--noise-seed", type=int, default=0, help="seed of the corruption noise")
    args = parser.parse_args()

    setup_logging()
    log = get_logger("robustness")
    for run_dir in map(Path, args.runs):
        cfg = load_config(run_dir / "config.yaml")
        seed_everything(cfg.seed)
        device = resolve_device(cfg.device)
        splits = build_train_val(cfg.data)
        model = build_model(cfg.model, splits.in_channels, cfg.data.image_size)
        load_checkpoint(run_dir / "checkpoints" / "best.pt", model)
        model.to(device)
        loader = make_loader(splits.val, 500, False, cfg.seed)

        results = {
            "protocol": "N1",
            "split": "val",
            "noise_seed": args.noise_seed,
            "gaussian": {},
            "salt_pepper": {},
        }
        for kind, levels in (("gaussian", GAUSSIAN), ("salt_pepper", SALT_PEPPER)):
            for level in levels:
                r = evaluate_corruption(model, loader, device, kind, level, args.noise_seed)
                results[kind][str(level)] = r
                log.info(
                    "%s %s=%.2f psnr=%.2f (input %.2f) ssim=%.4f (input %.4f)",
                    run_dir.name,
                    kind,
                    level,
                    r["psnr"],
                    r["input_psnr"],
                    r["ssim"],
                    r["input_ssim"],
                )
        (run_dir / "robustness_val.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
