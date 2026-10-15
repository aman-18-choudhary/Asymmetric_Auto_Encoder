# Phase 5: Core Experiments and Baseline Comparison

**Scope actually executed: 3 runs (seed 0 of each model), not the 9 runs in the approved plan.**
All statements below are single-seed; no variability estimate exists.

## A. Phase Summary
Train the baseline, a parameter-matched baseline and the asymmetric autoencoder under one protocol on CIFAR-10, evaluate reconstruction on validation and (once) on the official test set, and measure model cost.

## B. Objectives
Both architectures plus a parameter-matched baseline; PSNR/SSIM/MAE; efficiency; reconstruction grids; comparison plots; machine-readable results; no unsupported claims.

## C. Work Completed
- Metrics: MSE, MAE, PSNR (MAX=1.0, per image), SSIM (11x11 Gaussian, sigma 1.5; agrees with scikit-image within 5e-3 in a test).
- Efficiency: parameters, FLOPs (`torch.utils.flop_counter`), latency (CPU, MPS; batch 1 and 128).
- Parameter-matched baseline: channels `(104, 208, 416)`, 2,812,227 parameters vs 2,823,625 (-0.40%).
- Runs completed (seed 0, 30 epochs each): `baseline_cifar10`, `baseline_matched_cifar10`, `asymmetric_cifar10`.
- One test-set evaluation of the three best-validation checkpoints.
- Aggregation tables and figures; tests (105 total) for metrics and reporting.

## D. Architecture and Technical Decisions
- Protocol (identical for all models, frozen at commit `c670b16`): CIFAR-10, 45k/5k split (`split_seed` 0), Adam lr 1e-3, cosine schedule over 30 epochs, batch 128, no early stopping, seed 0, latent d=64, [0,1] pixel range.
- Model selection uses validation MSE only. The test set was evaluated once, after all training and validation analysis, and no checkpoint, hyperparameter or model choice was changed afterwards (`best.pt` hashes verified identical before and after).
- The parameter-matched baseline matches parameters, not compute (see efficiency).

## E. Files Added or Modified
Added: `evaluation/{metrics,efficiency,reporting}.py`, `utils/visualization.py`, `scripts/{evaluate,aggregate,run_core_experiments.sh}`, `configs/baseline_matched.yaml`, `tests/{test_metrics,test_reporting}.py`, `reports/figures/*.png`, `experiments/results/**` (histories, configs, summaries, evaluation JSON; checkpoints git-ignored), this report. Modified: `configs/{baseline,asymmetric}.yaml` (explicit cosine schedule), `pyproject.toml` (matplotlib; scikit-image in dev), `scripts/aggregate.py`, `evaluation/reporting.py` (no "±" for a single run).

## F. Commands Executed
```
bash scripts/run_core_experiments.sh                       # training (see I for what actually ran where)
python scripts/evaluate.py <run dirs> --split val
python scripts/evaluate.py <run dirs> --split test --allow-test    # once
python scripts/aggregate.py --seeds 0 --split val --no-grid; python scripts/aggregate.py --seeds 0 --split test
```
Efficiency was measured with the repository functions in `evaluation/efficiency.py` over the three `best.pt` checkpoints (results stored in each run's `efficiency.json`).

## G. Verification Results
- `pytest`: 105 passed; `ruff check` and `ruff format --check` clean (excluding markdown).
- Each run: 30/30 epochs, `summary.json`, `last.pt` at epoch 30, `best.pt` loads, saved `config.yaml` equals the frozen config, device MPS.
- Re-evaluated validation MSE from `best.pt` equals the recorded best to 6 decimals for all three runs.
- Test MSE differs from validation MSE by only 1.6e-5 to 2.4e-5 (about 0.2-0.3%) for each model, so there is no sign that checkpoint selection overfit the validation split.

## H. Experimental Results (seed 0 only)

| Model | Params | FP32 MB | MACs/img | Val MSE | Val SSIM | **Test MSE** | **Test MAE** | **Test PSNR (dB)** | **Test SSIM** |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 450,627 | 1.72 | 5.4 M | 0.007586 | 0.6491 | 0.007601 | 0.0615 | 21.81 | 0.6496 |
| baseline_matched | 2,812,227 | 10.73 | 52.1 M | 0.007486 | 0.6499 | 0.007510 | 0.0612 | 21.87 | 0.6503 |
| asymmetric | 2,823,625 | 10.77 | 279.9 M | 0.007660 | 0.6312 | 0.007683 | 0.0621 | 21.76 | 0.6318 |

Validation MAE/PSNR: baseline 0.0614/21.81, matched 0.0610/21.86, asymmetric 0.0619/21.75 (5,000 images). Test uses 10,000 images.

Efficiency and training cost (Apple M2, 8 GB, MPS; latency = median over 50 iterations, 10 warmup):

| Model | CPU b=1 ms | CPU b=128 ms | MPS b=1 ms | MPS b=128 ms | s/epoch (MPS) | Train time (30 ep) | Best epoch |
|---|---|---|---|---|---|---|---|
| baseline | 0.21 | 30.1 | 0.82 | 3.9 | 11.9 | 5.9 min | 30 |
| baseline_matched | 0.51 | 144.1 | 1.74 | 30.4 | 39.5 | 19.8 min | 14 |
| asymmetric | 3.06 | 398.0 | 2.80 | 53.0 | 63.0 | 31.5 min | 30 |

Parameters split (encoder / decoder): baseline 224,384 / 226,243; matched 1,403,024 / 1,409,203; asymmetric 2,773,888 / 49,737.
Figures: `reports/figures/{val_mse_curves,test_mse,test_psnr,test_ssim,val_mse,val_psnr,val_ssim,reconstructions_test}.png`. Tables: `experiments/results/core_comparison_{val,test}.md`, `core_comparison.json`.

**What the data show.** On test MSE, PSNR and SSIM the three models are within ~2.3% (MSE) of each other; ordering is matched < baseline < asymmetric. The asymmetric model is numerically 1.1% worse than the baseline and 2.3% worse than the matched baseline in MSE, and 0.018 lower in SSIM (about 2.7% relative to the baseline). The asymmetric model needs 5.4x the MACs of the matched baseline and 52x the baseline's, and is 3.4-14.3x slower than the baseline (1.6-6.0x slower than the matched baseline) depending on device and batch size. The 6.2x larger matched baseline improves MSE by only 1.2% over the small baseline.

**What they do not show.** With one seed per model these small differences cannot be called real or ranked statistically; the earlier 2-epoch pilot runs differed by several percent between seeds, so seed noise of this order is plausible but is not measured at 30 epochs. The reconstructions are visibly blurry for all models (`reconstructions_test.png`); the asymmetric model's early validation curve is also noisier (e.g. a spike at epoch 16) and still improving at epoch 30.

### Hypotheses from the approved blueprint (status after this phase)
| ID | Status |
|---|---|
| H1 (asymmetric no worse at equal latent size) | Not supported by this single-seed evidence (numerically slightly worse, notably in SSIM); not statistically rejected either. |
| H2 (smaller latent at equal quality) | Not tested (latent sweep not run). |
| H3 (noise robustness) | Not tested. |
| H4 (linear-probe quality) | Not tested. |
| H5 (collapse) | Not tested. |

## I. GitHub Activity
- Frozen protocol commit: `c670b16`. Runner cooldown commit `abf3bdf` (script only; `src/` and `configs/` unchanged since `c670b16`).
- Provenance of runs: `baseline` seed 0 and `baseline_matched` seed 0 (epochs 1-28) were trained on the Mac (MPS) in the first session; matched epochs 29-30 resumed on the same Mac; `asymmetric` seed 0 trained on the Mac. All three runs are therefore single-device (MPS).
- Final commit/push status: see the completion message; nothing in this section is claimed beyond what the remote shows.

## J. Issues Encountered
- Mac overheating forced a pause; training moved to Google Colab, which completed nine runs, but those artifacts were lost to a runtime reset and the Colab quota was exhausted. Seeds 1 and 2 were not re-run on the Mac by decision (3-experiment plan). No Colab-trained result is used anywhere in this report.
- An early version of the resume logic failed on MPS (fixed in Phase 4).
- Model-selection plateau: the matched baseline's best epoch (14) precedes the end of training, so `best.pt` and `last.pt` differ for that run only.

## K. Outstanding Issues and Limitations
- **Single seed.** No means, standard deviations or significance tests; the planned 3-seed comparison was not completed. Treat all results as preliminary.
- Only 3 of 9 planned runs; no multi-seed variance for timing either.
- Training times are Apple MPS with cooldown pauses and thermal effects; latency is specific to this machine, torch 2.14.1, 4 CPU threads. FLOPs count only convolutions and linear layers.
- Reconstruction metrics only; no noise-robustness (Phase 6), latent-space analysis or linear probe (Phase 7), so no claim about semantic representation quality.
- Asymmetric model may be under-trained at 30 epochs (best epoch = 30); not tested with longer training.
- Colab/CUDA execution and CUDA checkpoint loading were never verified.
- No CI, no GitHub PRs.

## L. Acceptance Criteria
| Criterion | Status |
|---|---|
| Both models have completed valid runs | PASS (seed 0 each, plus matched baseline) |
| Experimental settings documented | PASS |
| Results stored in a reproducible format | PASS (JSON/MD, configs, histories) |
| Comparisons use consistent evaluation data | PASS |
| No unsupported performance claims | PASS (single-seed limitation stated) |
| Three-seed statistical comparison (original plan) | FAIL / not completed |

## M. Next Phase
Phase 6 (noise robustness and ablations) is not started and awaits approval. It would reuse these checkpoints for the test-time corruption experiment (protocol N1) and would need additional training for the latent sweep and denoising training (N2).
