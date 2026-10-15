# Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning

A controlled study of whether an asymmetric autoencoder (heavy nonlinear encoder, tight latent
bottleneck, lightweight multiscale decoder) learns more compact and noise-robust representations
than a conventional symmetric convolutional autoencoder under matched training conditions.

> **Status: Phase 3 (asymmetric architecture implemented, not yet evaluated).** Both models exist
> and run; no comparison experiments have been performed, so no performance claim is made.
> The architecture and hypotheses are *proposals*; see
> [`docs/phase_reports/phase_00_research_blueprint.md`](docs/phase_reports/phase_00_research_blueprint.md).

## Setup

Requires Python >= 3.10 (developed on 3.11).

```bash
git clone https://github.com/aman-18-choudhary/Asymmetric_Auto_Encoder.git
cd Asymmetric_Auto_Encoder
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data

Datasets are downloaded on demand into `data/` (git-ignored). MNIST (~11 MB, debugging) and
CIFAR-10 (~163 MB, primary):

```bash
python scripts/download_data.py --name mnist
python scripts/download_data.py --name cifar10
```

Images are scaled to [0, 1] (no mean/std normalisation); MNIST is resized to 32x32. The
train/validation split is deterministic and depends only on `data.split_seed` (default 0), so all
models share it: CIFAR-10 gives 45,000 train / 5,000 val, and the official 10,000 test images are
kept for final reporting only. Labels are never used for training.

## Debug runs (baseline or asymmetric, set by the config)

```bash
python scripts/train_debug.py --config configs/mnist_debug.yaml   # ~30 s on Apple M2 (MPS)
python scripts/train_debug.py --config configs/baseline.yaml --epochs 1 --max-batches 50
```

The same script runs Model B with `--config configs/asymmetric.yaml` (set `model.type`).
Asymmetric options: `enc_channels`, `dec_channels`, `refine_steps` (shared-weight refinement),
`multiscale`, `scale_weights`, `latent_norm`. Architecture details are in
`src/asymmetric_autoencoder/models/asymmetric_autoencoder.py`.

These are debugging runs, not reported experiments. `configs/baseline.yaml` is the CIFAR-10
configuration for the later full experiments (Phase 5).

## Development

```bash
pytest              # run tests
ruff check .        # lint
ruff format --check .   # formatting check
```

Configuration lives in `configs/` (YAML, validated by `asymmetric_autoencoder.config`; unknown
keys are rejected). Seeding and device selection: `asymmetric_autoencoder.utils.reproducibility`.
Logging: `asymmetric_autoencoder.utils.logging`.

## Git workflow

`main` (stable) <- `develop` (integration) <- `feature/phase-N-*`. Conventional Commits
(`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `exp:`). Datasets and checkpoints are never committed.

## Reproducibility notes

Seeds are set for Python, NumPy and PyTorch. Some MPS/CUDA kernels are non-deterministic;
`deterministic: true` requests deterministic algorithms and may raise for unsupported ops.
