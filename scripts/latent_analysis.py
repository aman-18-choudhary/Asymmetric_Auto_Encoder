"""Inference-only latent analysis of trained autoencoders (Phase 7).

No autoencoder is trained. PCA bases and linear probes are fitted on the TRAIN split features
and scored on the VALIDATION split; the official test split is never loaded by this script.
"""

import argparse
import json
from pathlib import Path

import torch

from asymmetric_autoencoder.config import load_config
from asymmetric_autoencoder.data.datasets import build_train_val, make_loader
from asymmetric_autoencoder.evaluation import latent_analysis as la
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.training.checkpoints import load_checkpoint
from asymmetric_autoencoder.utils.logging import get_logger, setup_logging
from asymmetric_autoencoder.utils.reproducibility import resolve_device, seed_everything
from asymmetric_autoencoder.utils.visualization import bar_plot, latent_scatter, line_plot

MODELS = {
    "baseline": "baseline_cifar10",
    "baseline_matched": "baseline_matched_cifar10",
    "asymmetric": "asymmetric_cifar10",
}
KS = [0, 1, 2, 4, 8, 16, 32, 48, 64]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads (keeps the Mac cool)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    setup_logging()
    log = get_logger("latent_analysis")
    root, fig_dir = Path(args.results_dir), Path(args.fig_dir)
    cfg0 = load_config(root / f"{MODELS['baseline']}_seed{args.seed}" / "config.yaml")
    seed_everything(args.seed)
    device = resolve_device(cfg0.device)
    splits = build_train_val(cfg0.data)
    train_loader = make_loader(splits.train, 500, False, args.seed)
    val_loader = make_loader(splits.val, 500, False, args.seed)
    class_names = list(splits.train.dataset.classes)

    log.info("extracting raw pixels for the pixel controls")
    px_tr, y_tr = la.extract_flat_images(train_loader)
    px_va, y_va = la.extract_flat_images(val_loader)
    probes: dict[str, dict] = {}
    probes["raw_pixels (3072-d)"] = la.fit_linear_probe(px_tr, y_tr, px_va, y_va)
    pca_px = la.fit_pca(px_tr)
    probes["raw_pixels PCA-64"] = la.fit_linear_probe(
        la.project(px_tr, pca_px, 64), y_tr, la.project(px_va, pca_px, 64), y_va
    )
    del px_tr, px_va
    log.info(
        "pixel controls done: %s", {k: round(v["eval_accuracy"], 4) for k, v in probes.items()}
    )

    results: dict[str, dict] = {}
    for name, exp in MODELS.items():
        run = root / f"{exp}_seed{args.seed}"
        cfg = load_config(run / "config.yaml")
        model = build_model(cfg.model, splits.in_channels, cfg.data.image_size)
        load_checkpoint(run / "checkpoints" / "best.pt", model)
        model.to(device).eval()
        z_tr, y_tr_ = la.extract_latents(model, train_loader, device)
        z_va, y_va_ = la.extract_latents(model, val_loader, device)
        stats_tr = la.latent_statistics(z_tr)
        stats_va = la.latent_statistics(z_va)
        basis = la.fit_pca(z_tr)
        trunc = la.truncation_curve(
            model, val_loader, device, basis, [k for k in KS if k <= z_tr.shape[1]]
        )
        probe = la.fit_linear_probe(z_tr, y_tr_, z_va, y_va_)
        probes[f"{name} (trained)"] = probe

        # control: same architecture, random initialisation (seed fixed), no training
        seed_everything(args.seed)
        rand = build_model(cfg.model, splits.in_channels, cfg.data.image_size).to(device).eval()
        zr_tr, _ = la.extract_latents(rand, train_loader, device)
        zr_va, _ = la.extract_latents(rand, val_loader, device)
        probes[f"{name} (random init)"] = la.fit_linear_probe(zr_tr, y_tr_, zr_va, y_va_)
        rstats = la.latent_statistics(zr_tr)

        results[name] = {
            "run": run.name,
            "stats_train": stats_tr,
            "stats_val": stats_va,
            "random_init_stats_train": rstats,
            "truncation_val_mse": {str(k): v for k, v in trunc.items()},
            "probe": probe,
        }
        (run / "latent_analysis_val.json").write_text(json.dumps(results[name], indent=2))
        latent_scatter(
            la.project(z_va, basis, 2),
            y_va_,
            fig_dir / f"latent_pca_{name}.png",
            f"{name}: validation latents on the first two PCs (illustrative only)",
            class_names,
        )
        log.info(
            "%s: eff.rank=%.1f probe=%.4f", name, stats_tr["effective_rank"], probe["eval_accuracy"]
        )

    out = {
        "results": results,
        "probes": probes,
        "split": "train (fit) / val (score)",
        "seed": args.seed,
    }
    (root / "latent_analysis_val.json").write_text(json.dumps(out, indent=2))

    cum = {n: r["stats_train"]["explained_variance_cumulative"] for n, r in results.items()}
    line_plot(
        {n: (list(range(1, len(v) + 1)), v) for n, v in cum.items()},
        fig_dir / "latent_explained_variance.png",
        "Cumulative explained variance of trained latents (train split)",
        "number of principal components",
        "cumulative variance fraction",
    )
    line_plot(
        {
            n: ([int(k) for k in r["truncation_val_mse"]], list(r["truncation_val_mse"].values()))
            for n, r in results.items()
        },
        fig_dir / "latent_truncation_mse.png",
        "Validation MSE when decoding latents truncated to k PCs (post-hoc, no retraining)",
        "k retained principal components",
        "validation MSE",
        logy=True,
    )
    bar_plot(
        {k: v["eval_accuracy"] for k, v in probes.items()},
        fig_dir / "latent_probe_accuracy.png",
        "Linear-probe accuracy on validation (probe fit on train features)",
        "accuracy",
        errors={k: v["eval_accuracy_stderr"] for k, v in probes.items()},
        hline=0.1,
        hline_label="chance",
    )

    lines = [
        "| Representation | features | probe val accuracy ± s.e. | probe train acc | L2 |",
        "|---|---|---|---|---|",
    ]
    for k, v in probes.items():
        lines.append(
            f"| {k} | {v['n_features']} | {v['eval_accuracy']:.4f} ± "
            f"{v['eval_accuracy_stderr']:.4f} | {v['train_accuracy']:.4f} | {v['l2']:g} |"
        )
    lines += [
        "",
        "| Model | eff. rank | participation ratio | dims for 90/95/99% var | "
        "low-var dims | mean dim std | random-init eff. rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, r in results.items():
        s = r["stats_train"]
        lines.append(
            f"| {n} | {s['effective_rank']:.1f} | {s['participation_ratio']:.1f} | "
            f"{s['dims_for_90pct_variance']}/{s['dims_for_95pct_variance']}/"
            f"{s['dims_for_99pct_variance']} | {s['n_low_variance_dims']} | "
            f"{s['mean_dim_std']:.3f} | {r['random_init_stats_train']['effective_rank']:.1f} |"
        )
    ks = [k for k in KS if k <= 64]
    lines += [
        "",
        "| Model | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for n, r in results.items():
        lines.append(
            f"| {n} | " + " | ".join(f"{r['truncation_val_mse'][str(k)]:.5f}" for k in ks) + " |"
        )
    (root / "latent_analysis_val.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
