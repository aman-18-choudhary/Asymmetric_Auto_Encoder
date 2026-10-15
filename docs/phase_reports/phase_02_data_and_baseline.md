# Phase 2: Dataset Pipeline and Baseline Autoencoder

## A. Phase Summary
Implement the MNIST/CIFAR-10 data pipeline with reproducible splits and the conventional convolutional autoencoder (Model A), with tests and a small-batch overfit check. The asymmetric model and large experiments are out of scope.

## B. Objectives
Dataset download/loading; deterministic train/val split; preprocessing; baseline AE; shape and forward/backward tests; tiny-batch overfit; minimal training loop; documentation.

## C. Work Completed
- `data/datasets.py`: registry (mnist, fashion_mnist, cifar10), `build_transform`, `split_indices`, `build_splits`, seeded `make_loader`.
- `models/baseline_autoencoder.py`: `ConvAutoencoder` (`encode`/`decode`/`forward`), `count_parameters`.
- `training/simple.py`: minimal Adam+MSE loop and `evaluate_mse` (replaced by Phase 4 infrastructure).
- `scripts/download_data.py`, `scripts/train_baseline.py` (thin).
- Config extended: `data.split_seed`, `model`, `train` sections; `configs/baseline.yaml` (CIFAR-10), `configs/mnist_debug.yaml`.
- Tests: data, baseline model, config additions. README updated.

## D. Architecture and Technical Decisions
- Baseline: 3x [Conv 3x3 stride 2 + ReLU] (channels 32/64/128) -> Flatten -> Linear -> d; mirrored Linear + 3 transposed convs + Sigmoid; plain MSE. Matches the approved blueprint (Model A).
- Preprocessing: `ToTensor` only, so data lie in [0,1], matching the sigmoid output and the PSNR/SSIM `data_range=1.0`. No mean/std normalisation.
- MNIST resized 28->32 so debug runs use the identical architecture (1 channel instead of 3).
- Split depends only on `(n, val_fraction, split_seed)`, independent of the training seed, so every model gets the same split. Test set is the official test split, untouched until final evaluation.
- Parameter-count reference (verified): d=64, 3 channels: 450,627; d=32, 1 channel: 318,369.
- Downloads are explicit (`download=False` by default; no network access in tests).
- Not changed: hypotheses, architecture proposals and protocols in the approved Phase 0 report.

## E. Files Added or Modified
Added: `src/asymmetric_autoencoder/data/datasets.py`, `models/baseline_autoencoder.py`, `training/simple.py`, `scripts/download_data.py`, `scripts/train_baseline.py`, `configs/baseline.yaml`, `configs/mnist_debug.yaml`, `tests/test_data.py`, `tests/test_baseline_model.py`, this report.
Modified: `config.py`, `tests/test_config.py`, `README.md`.

## F. Commands Executed
```
python scripts/download_data.py --name mnist
python scripts/download_data.py --name cifar10      # user-approved download, ~163 MB
python scripts/train_baseline.py --config configs/mnist_debug.yaml
pytest -q; ruff check .; ruff format --check .
git push -u origin feature/phase-2-data-baseline
```

## G. Verification Results
- `pytest`: 47 passed; `ruff check` and `ruff format` clean.
- Real CIFAR-10: split 45,000 / 5,000 / 10,000 test; train/val disjoint; split repeatable across calls; batch `[128,3,32,32]` float32 in [0,1]; baseline output shape equals input shape.
- Real CIFAR-10 16-image batch overfit (400 Adam steps, lr 2e-3, one seed, MPS): MSE 0.0721 -> 0.00031. Debugging check only, not a performance result.
- MNIST debug run (3 epochs, MPS, latent 32, seed 0): train MSE 0.0356 / 0.0061 / 0.0042; validation MSE 0.0038. Single seed, 3 epochs: a pipeline sanity check, not a reported experiment.
- Gradient flow: all parameters receive finite gradients (unit test).
- Not tested: CUDA; multi-worker DataLoader; `deterministic=True` on MPS; CIFAR-10 full-length training.

## H. Experimental Results
No experiment results are claimed. The numbers in G are debugging checks (single seed, tiny budgets).

## I. GitHub Activity
- Branch `feature/phase-2-data-baseline`, created from `develop` at `34d2fd8`.
- Commits: `7ebf528` feat: extend config with split seed, model and train sections; `5865b88` feat: add MNIST/CIFAR-10 pipeline with deterministic train/val split; `b99ef7e` feat: add baseline conv autoencoder, minimal training loop and run scripts; `3106f8a` test: add data pipeline and baseline model tests; `560cf1a` docs: document data preparation and baseline execution.
- This report is committed afterwards; push status is given in the completion message after remote verification.
- Not merged into `develop` (awaiting your review). No force-push. No PR (no `gh`).
- Phase 1 integration: `develop` = `34d2fd8` (`--no-ff` merge), verified on the remote.

## J. Issues Encountered
- CIFAR-10 download from the source server was slow (~40 min) but completed.
- `ruff --fix` reordered one import; harmless.

## K. Outstanding Issues
- No CI workflow; no GitHub PRs (no `gh`).
- `training/simple.py` has no checkpoints/scheduler (Phase 4).
- `ConvAutoencoder` has no FLOPs/latency measurement yet (Phase 5/6).
- Latent is unnormalised linear output; collapse diagnostics come in Phase 7.
- MPS non-determinism is undocumented in numbers; same-seed run-to-run drift not measured.
- Pure-Python 3.10/3.12 untested.

## L. Acceptance Criteria
| Criterion | Status |
|---|---|
| Data loading works | PASS (MNIST + CIFAR-10) |
| Batch dimensions correct | PASS |
| Model output matches target dimensions | PASS |
| Forward and backward passes succeed | PASS |
| Small training run completes | PASS (MNIST 3 epochs, MPS) |
| Tests pass | PASS (47) |
| Baseline overfits a small batch | PASS (unit test on synthetic data; real CIFAR-10 batch) |

## M. Next Phase
Phase 3: the asymmetric architecture (encoder, bottleneck, lightweight multiscale decoder, multiscale loss) with parameter-count comparison. Not started.
