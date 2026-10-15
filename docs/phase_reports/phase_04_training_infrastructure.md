# Phase 4: Training Infrastructure

## A. Phase Summary
One reusable training pipeline for both autoencoders: configurable optimizer/scheduler, per-epoch validation, history, early stopping, atomic checkpoints, exact resume, and per-run configuration/environment snapshots. No comparison experiments were run.

## B. Objectives
Trainer; optimizer and scheduler configuration; checkpoint save/load; validation; history; early stopping; CPU/accelerator support; saved configs; both models on the same infrastructure.

## C. Work Completed
- `training/trainer.py`: `Trainer`, `evaluate`, `run_dir_for`.
- `training/checkpoints.py`: atomic `save_checkpoint`, `load_checkpoint` (`weights_only=True`).
- `utils/reproducibility.py`: `get_rng_state` / `set_rng_state` (Python, NumPy, torch, CUDA, MPS).
- `config.py`: `train.optimizer`, `weight_decay`, `scheduler`, `early_stopping_patience`, `early_stopping_min_delta`, `output_dir`.
- `scripts/train.py` replaces `train_debug.py`; the Phase 2 `training/simple.py` loop was removed.
- 19 new tests (93 total).

## D. Architecture and Technical Decisions
- Selection criterion is `val_mse`: final-resolution MSE on the clean input, comparable across models. The training objective differs per model (multiscale vs plain MSE), so `train_loss`/`val_loss` are recorded but never used to compare models.
- Exact resume: the shuffling generator is re-seeded each epoch (`seed + epoch`), and RNG states plus optimizer/scheduler/history/counters are in `last.pt`.
- Safety: atomic checkpoint writes; `weights_only=True` loading; refuse to start into a directory containing `last.pt` unless resuming (prevents silent overwrite); non-finite loss raises `FloatingPointError`.
- Cosine scheduler is stepped per epoch with `T_max = epochs`; a resumed run must use the same `epochs` to keep the same schedule.
- Not implemented (deliberately): mixed precision, gradient clipping, multi-GPU, StepLR.

## E. Files Added or Modified
Added: `training/trainer.py`, `training/checkpoints.py`, `tests/test_training.py`, `docs/phase_reports/phase_04_training_infrastructure.md`. Modified: `config.py`, `utils/reproducibility.py`, `tests/test_data.py` (old minimal-loop test moved into trainer tests), `README.md`. Renamed/rewritten: `scripts/train_debug.py` -> `scripts/train.py`. Removed: `training/simple.py`.

## F. Commands Executed
`pytest -q`; `ruff check .` / `ruff format .`; `python scripts/train.py --config configs/{baseline,asymmetric}.yaml --epochs 3 --max-batches 40 --run-dir <scratch>`; resume and no-overwrite smoke runs; a 4-epoch full vs 2+2-epoch resumed comparison on MPS.

## G. Verification Results
- `pytest`: 93 passed; `ruff check` clean.
- Both models train through the same `Trainer`; history has one complete record per epoch; run directory contains config, environment, history, summary, `last.pt`, `best.pt`.
- Best checkpoint reproduces the recorded best `val_mse`.
- Resume (CPU, unit tests, both models, cosine scheduler): resumed run's per-epoch train loss, val MSE, learning rate and final weights are bit-identical to the uninterrupted run.
- Resume (MPS, script, asymmetric, 30 batches/epoch, 4 epochs vs 2+2): all 4 epochs bit-identical (train loss and val MSE). One configuration, one seed.
- Early stopping, cosine decay, overwrite refusal, resume-after-completion no-op, NaN guard: tested.
- Smoke runs (3 epochs, 40 batches, seed 0, MPS, not experiments): baseline val_mse 0.0348 -> 0.0243; asymmetric 0.0293 -> 0.0206.

## H. Experimental Results
None claimed. The smoke-run numbers above use a 40-batch cap, one seed, and unequal model sizes.

## I. GitHub Activity
Branch `feature/phase-4-training` from `develop` at `7b88bf6` (Phase 3 merge, verified on remote). Commits and push status are given in the completion message after remote verification. Not merged.

## J. Issues Encountered
- First MPS resume attempt failed: `map_location=device` moved the stored RNG tensors onto MPS and NumPy could not read them. CPU-only tests had not caught it. Fixed (load checkpoints on CPU; `set_rng_state` copies to host) and a regression test that runs when an accelerator is available was added.
- A shell mistake (unquoted variable) silently skipped my first MPS exactness run; re-run correctly.

## K. Outstanding Issues
- Exact resume verified on CPU (tests) and one MPS configuration only; CUDA untested; multi-worker data loading untested.
- `deterministic=True` still untested on MPS.
- Resuming with a different `epochs` under a cosine schedule changes the schedule (documented, not guarded).
- No FLOPs/latency measurement, no parameter-matched baseline, no metrics beyond MSE yet (Phase 5).
- No CI, no PRs (`gh` not installed).

## L. Acceptance Criteria
| Criterion | Status |
|---|---|
| Both models use the same training infrastructure | PASS |
| Training and validation losses are recorded | PASS |
| Checkpoints can be restored | PASS |
| Training can be resumed | PASS (CPU tests; one MPS check) |
| Results reproducible within documented limits | PASS (same-device exact resume; CPU vs MPS differ, see Phase 2) |
| Experiment configurations saved with results | PASS |

## M. Next Phase
Phase 5: core experiments: train both models with the same protocol and 3 seeds, a parameter-matched baseline, PSNR/SSIM/MAE, efficiency measurements, reconstruction grids and comparison plots. Not started.
