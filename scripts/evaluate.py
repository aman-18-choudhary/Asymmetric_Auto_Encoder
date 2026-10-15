"""Evaluate trained runs (best checkpoint) and measure model cost.

Validation metrics are always allowed. The official TEST split requires --allow-test: it is
reserved for the single final evaluation after all training decisions are frozen.
"""

import argparse
import json
from pathlib import Path

import torch

from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.data.datasets import build_test_set, build_train_val, make_loader
from asymmetric_autoencoder.evaluation.efficiency import count_flops, measure_latency
from asymmetric_autoencoder.evaluation.metrics import evaluate_reconstruction
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.models.summary import count_parameters, parameter_breakdown
from asymmetric_autoencoder.training.checkpoints import load_checkpoint
from asymmetric_autoencoder.utils.logging import get_logger, setup_logging
from asymmetric_autoencoder.utils.reproducibility import resolve_device, seed_everything

CPU = torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true", help="required for --split test")
    parser.add_argument("--efficiency", action="store_true", help="also write efficiency.json")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("--split test requires --allow-test (final evaluation only)")

    setup_logging()
    log = get_logger("evaluate")
    for run_dir in map(Path, args.runs):
        cfg = load_config(run_dir / "config.yaml")
        seed_everything(cfg.seed)
        device = resolve_device(cfg.device)
        splits = build_train_val(cfg.data)
        model = build_model(cfg.model, splits.in_channels, cfg.data.image_size)
        load_checkpoint(run_dir / "checkpoints" / f"{args.checkpoint}.pt", model)
        model.to(device)

        dataset = splits.val if args.split == "val" else build_test_set(cfg.data)
        loader = make_loader(dataset, 500, False, cfg.seed)
        metrics = evaluate_reconstruction(model, loader, device)
        metrics["checkpoint"] = args.checkpoint
        (run_dir / f"eval_{args.split}.json").write_text(json.dumps(metrics, indent=2))
        log.info("%s %s: %s", run_dir.name, args.split, json.dumps(metrics))

        if args.efficiency:
            shape = (splits.in_channels, cfg.data.image_size, cfg.data.image_size)
            eff = {
                "parameters": count_parameters(model),
                "parameter_breakdown": parameter_breakdown(model),
                "latent_dim": cfg.model.latent_dim,
                "flops": count_flops(model.to(CPU), shape),
                "latency": {
                    "cpu_batch1": measure_latency(model, shape, CPU, 1),
                    "cpu_batch128": measure_latency(model, shape, CPU, 128, iters=20),
                },
            }
            if device.type != "cpu":
                eff["latency"][f"{device.type}_batch1"] = measure_latency(model, shape, device, 1)
                eff["latency"][f"{device.type}_batch128"] = measure_latency(
                    model, shape, device, 128
                )
            (run_dir / "efficiency.json").write_text(json.dumps(eff, indent=2))
            log.info("%s efficiency: %s", run_dir.name, json.dumps(eff))


if __name__ == "__main__":
    main()
