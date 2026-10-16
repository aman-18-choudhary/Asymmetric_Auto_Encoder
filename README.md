# Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning

A controlled study of whether an asymmetric autoencoder (heavy nonlinear encoder, tight latent
bottleneck, lightweight multiscale decoder) learns more compact and noise-robust representations
than a conventional symmetric convolutional autoencoder under matched training conditions.

## Project status

Final report: **[`reports/final_report.md`](reports/final_report.md)** (per-phase details in
[`docs/phase_reports/`](docs/phase_reports/)). Phase 9 (final repository audit) is **not** done.

**Completed**
- Reproducible pipeline: deterministic 45k/5k split, shared `Trainer` with checkpoint/resume, 121 tests.
- Three models implemented: baseline, parameter-matched baseline, asymmetric autoencoder.
- One 30-epoch CIFAR-10 training run per model (seed 0), one final test-set evaluation, efficiency
  measurements, input-noise robustness (validation split) and an inference-only latent analysis.

**Results so far are single-seed evidence** (one training run per model; no variability measured):

| Model | Params | MACs/img | Test MSE | Test PSNR (dB) | Test SSIM | PSNR drop at noise σ=0.3 (val) | Linear-probe acc. (val) |
|---|---|---|---|---|---|---|---|
| baseline | 450,627 | 5.4 M | 0.007601 | 21.81 | 0.6496 | 2.25 dB | 37.9% |
| baseline_matched | 2,812,227 | 52.1 M | 0.007510 | 21.87 | 0.6503 | 2.37 dB | 37.5% |
| asymmetric | 2,823,625 | 279.9 M | 0.007683 | 21.76 | 0.6318 | 1.75 dB | 37.9% |

The asymmetric model did not beat the baselines on reconstruction or linear-probe accuracy and
costs several times more compute. Its noise-robustness advantage is only directionally supported
(H3), not established. No differences here are statistically assessed.

**Incomplete / not done**
- Seeds 1 and 2 (6 of the 9 planned runs were not completed; Colab results were lost).
- Latent-dimension sweep: hypothesis **H2 is not fully tested**.
- Component ablations, denoising training (N2), t-SNE/UMAP, CUDA validation.
- Phase 9 (final repository audit and presentation preparation).

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

**Warning:** the seed-0 result files (configs, histories, summaries, evaluation JSON, logs; not
checkpoints) are tracked in Git, so retraining with the default run directories overwrites them in
place (and appends to `train.log`); re-runs are not bit-identical to the committed numbers. Back up
`experiments/results/` first, or see "Reproducibility" in the final report for writing to a separate
directory.

Training options (`train:` section): `optimizer` (adam|adamw), `weight_decay`, `scheduler`
(none|cosine), `early_stopping_patience`, `early_stopping_min_delta`, `output_dir`.
Not implemented: mixed precision, gradient clipping, multi-GPU.

## Interactive demo (local, CPU only)

A Streamlit app for exploring the three trained models and the saved results. It is read-only: it
never trains, never writes to `experiments/results/` or `reports/figures/`, and never loads the
official test split (validation images are taken from the five training batches only).

```bash
pip install -e ".[demo]"                    # adds streamlit and pandas
python scripts/build_demo_assets.py         # once, inference only (about 30 s)
scripts/run_demo.sh                         # localhost only, usage statistics off
```

**Prerequisites for `scripts/build_demo_assets.py`.** The CIFAR-10 *training* batches must already
exist locally in `data/cifar-10-batches-py/` (`data_batch_1` ... `data_batch_5` and `batches.meta`), and
the three checkpoints must exist. The script only reads those files; it never opens `test_batch`
and does not download anything. Note that the repository's `scripts/download_data.py` fetches the
dataset through torchvision, whose integrity check also hashes `test_batch` (and downloads files if
anything is missing). If you want to avoid even that checksum read, obtain the training batches
yourself instead of running the download script.

**Privacy and local-only access.** `.streamlit/config.toml` disables Streamlit's anonymous usage
statistics. `scripts/run_demo.sh` also binds the server to `localhost` so other devices on your
network cannot reach it. A plain `streamlit run demo/app.py` does **not** restrict the address (the
server address is not set in the shared config, so that Community Cloud can host the app); if you
launch it that way locally, pass `--server.address localhost --browser.gatherUsageStats false`.
Asset generation writes only to `demo/assets/` (the path is resolved and anything under
`experiments/`, `reports/` or `data/`, or outside the project, is rejected).

**Public deployment (Streamlit Community Cloud).** Settings: repository
`aman-18-choudhary/Asymmetric_Auto_Encoder`, branch `feature/demo-streamlit`, main file
`demo/app.py`, Python 3.11 (Advanced settings). Cloud installs `requirements.txt` (it ranks above
`pyproject.toml`) and the app adds `src/` to its import path, so no package install is needed. The
checkpoints and `demo/assets/` are git-ignored, so a hosted copy of this repository shows the
**Results tab only**; the Reconstruct, Robustness and Latent tabs display a notice and need the
checkpoints and assets (run the demo locally).

Tabs: **Results** (saved metrics, efficiency, robustness and latent analysis; saved test numbers are
labelled as the one-time final evaluation), **Reconstruct** (validation samples through all three
models, with the asymmetric model's multiscale outputs), **Robustness (N1)** (evaluation-only input
corruption, scored against the clean image) and **Latent** (post-hoc PCA truncation, a 2-D projection
and interpolation: exploratory visualisations, not evidence of semantic structure).

The checkpoints (`experiments/results/*/checkpoints/best.pt`) and the generated `demo/assets/`
are git-ignored, so after a fresh clone the Results tab works immediately while the other tabs
need the trained checkpoints (see the final report for how to reproduce them). All results come from
one training run per model (seed 0); live numbers are computed on CPU and can differ slightly from the
saved Apple-MPS results.

## Development

```bash
pytest              # run tests
ruff check .        # lint
ruff format --check src scripts tests   # formatting check
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
