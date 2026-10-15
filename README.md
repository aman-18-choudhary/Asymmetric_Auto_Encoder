# Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning

A controlled study of whether an asymmetric autoencoder (heavy nonlinear encoder, tight latent
bottleneck, lightweight multiscale decoder) learns more compact and noise-robust representations
than a conventional symmetric convolutional autoencoder under matched training conditions.

> **Status: Phase 4 (shared training infrastructure).** Both models exist and train through one
> pipeline; no comparison experiments have been performed, so no performance claim is made.
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

## Training

One script and one `Trainer` serve both models; the model is chosen by `model.type` in the config.

```bash
python scripts/train.py --config configs/mnist_debug.yaml                  # fast debug (baseline)
python scripts/train.py --config configs/baseline.yaml  --seed 0           # Model A, CIFAR-10
python scripts/train.py --config configs/asymmetric.yaml --seed 0          # Model B, CIFAR-10
python scripts/train.py --config configs/asymmetric.yaml --resume          # continue an interrupted run
```

Useful flags: `--epochs N`, `--max-batches N` (debugging only), `--run-dir PATH`, `--seed S`.
Each run writes to `experiments/results/<experiment_name>_seed<seed>/` (override with `--run-dir`):

| File | Content |
|---|---|
| `config.yaml`, `environment.json` | exact configuration and library/device versions |
| `history.json` | per-epoch `train_loss`, `val_loss`, `val_mse`, `lr`, `seconds` |
| `summary.json` | best epoch, best `val_mse`, parameters, device, early-stop flag |
| `checkpoints/last.pt`, `best.pt` | resume state / best-validation weights (git-ignored) |

`val_mse` is the final-resolution MSE against the clean input and is the only loss comparable
across models; `train_loss`/`val_loss` are each model's own objective (plain MSE for the baseline,
weighted multiscale MSE for the asymmetric model). Best checkpoint and early stopping use `val_mse`.
Starting a run in a directory that already has `last.pt` fails unless `--resume` is given.
Resume restores model, optimizer, scheduler, history and RNG state and reproduces an
uninterrupted run exactly on the same device (tested on CPU, verified once on MPS).

Training options (`train:` section): `optimizer` (adam|adamw), `weight_decay`, `scheduler`
(none|cosine), `early_stopping_patience`, `early_stopping_min_delta`, `output_dir`.
Not implemented: mixed precision, gradient clipping, multi-GPU.

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
