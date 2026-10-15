# Phase 6: Noise Robustness and Ablations (PARTIAL: protocol N1 only)

**Executed:** test-time corruption robustness (N1) for the three existing seed-0 checkpoints.
**Not executed:** latent-dimension ablation, component ablations, denoising training (N2). These need
new training runs and are awaiting a scope decision. All results are single-seed.

## A. Phase Summary
Evaluate how each trained model's reconstruction degrades when its input is corrupted, scoring against the clean image.

## B. Objectives (from the approved plan)
Corruption evaluation (done); consistent conditions (done); latent-dimension ablation (not done); component ablations (not done); clean-target vs noisy-target protocols kept distinct (N1 only so far).

## C. Work Completed
- `evaluation/robustness.py`: `corrupt` (Gaussian, salt-and-pepper) and `evaluate_corruption`.
- `scripts/robustness.py`, `scripts/robustness_report.py`, `utils/visualization.robustness_curves`.
- `tests/test_robustness.py` (7 tests; 112 total with the earlier suite).
- Results: `robustness_val.json` per run, `experiments/results/robustness_val.md`, four figures `reports/figures/robustness_*`.

## D. Technical Decisions
- **Protocol N1:** models trained on clean images; input corrupted at evaluation; every metric computed against the clean image. No model was retrained.
- **Validation split only.** The official test split was used once in Phase 5 and is not reused here.
- **Deterministic corruption:** noise for batch `i` is drawn from a generator seeded with `(noise_seed, i)`, so all models see exactly the same corrupted images (checked: identical reference metrics across models).
- Gaussian: additive N(0, sigma^2) then clipped to [0,1], sigma in {0, 0.05, 0.1, 0.2, 0.3}. Salt-and-pepper: fraction p in {0.02, 0.05, 0.1} of pixel positions (all channels) set to 0 or 1 with equal probability.
- The corrupted input is also scored against the clean image (`input_*`), the score of a model that returns its input unchanged.

## E. Files
Added: `src/.../evaluation/robustness.py`, `scripts/robustness.py`, `scripts/robustness_report.py`, `tests/test_robustness.py`, this report, results and figures listed in C. Modified: `utils/visualization.py` (new function only).

## F. Commands
```
python scripts/robustness.py experiments/results/{baseline,baseline_matched,asymmetric}_cifar10_seed0
python scripts/robustness_report.py
```

## G. Verification
- `pytest`: 112 passed; `ruff check` clean.
- Corruption tests: identity at level 0, determinism, [0,1] range, Gaussian std within 5% of sigma, salt-and-pepper fraction within 1 point of p, channel consistency.
- At sigma = 0 the robustness evaluation reproduces `eval_val.json` MSE exactly for all three models.
- Evaluation sampling noise (4 corruption seeds, same models): PSNR std 0.001-0.005 dB, far below the between-model gaps below.

## H. Results (validation, 5,000 images, seed-0 models, PSNR dB / SSIM)

Gaussian noise:
| sigma | corrupted input | baseline | baseline_matched | asymmetric |
|---|---|---|---|---|
| 0 | identical | 21.81 / 0.649 | 21.86 / 0.650 | 21.75 / 0.631 |
| 0.05 | 26.19 / 0.847 | 21.75 / 0.646 | 21.79 / 0.646 | 21.70 / 0.629 |
| 0.1 | 20.33 / 0.655 | 21.54 / 0.636 | 21.56 / 0.636 | 21.55 / 0.623 |
| 0.2 | 14.80 / 0.402 | 20.70 / 0.601 | 20.69 / 0.600 | 20.94 / 0.599 |
| 0.3 | 11.94 / 0.268 | 19.56 / 0.552 | 19.49 / 0.550 | 20.00 / 0.564 |

Salt-and-pepper:
| p | corrupted input | baseline | baseline_matched | asymmetric |
|---|---|---|---|---|
| 0.02 | 22.20 / 0.817 | 21.47 / 0.631 | 21.47 / 0.629 | 21.48 / 0.617 |
| 0.05 | 18.11 / 0.639 | 20.98 / 0.606 | 20.91 / 0.602 | 21.05 / 0.597 |
| 0.1 | 15.07 / 0.467 | 20.21 / 0.567 | 20.09 / 0.562 | 20.36 / 0.565 |

PSNR drop from clean to sigma = 0.3: baseline 2.25 dB, matched 2.37 dB, asymmetric 1.75 dB. At salt-and-pepper p = 0.1: 1.60 / 1.78 / 1.40 dB.

## I. Interpretation (preliminary)
- All three models reconstruct a blurry, low-pass version of the image through a 64-dimensional bottleneck, so noise is attenuated structurally: above sigma ~0.1 every model scores far above the noisy input itself (e.g. 19.5-20.0 dB vs 11.9 dB at sigma 0.3). This is not specific to the asymmetric design.
- The asymmetric model degrades more slowly (smaller PSNR drop; higher PSNR from sigma >= 0.2; higher SSIM at sigma 0.3). Its advantage is 0.4-0.5 dB at sigma 0.3 against baselines. This is directionally consistent with H3.
- Caveats: (1) single training seed, so training-seed variance is unmeasured; (2) the asymmetric model starts lower (clean PSNR 21.75 vs 21.81-21.86 dB, SSIM 0.631 vs 0.65) and a model that discards more detail also has less to lose, so a smaller drop is not by itself better denoising; (3) SSIM favours the baselines at low noise and the ordering reverses only at sigma 0.3; (4) these models were never trained to denoise (N2 not run).
- H3 status: tentatively supported in direction, not established.

## J. Issues Encountered
- The uncorrupted input hits the 100 dB PSNR floor and made the first figure unreadable; that point is now omitted from plots and shown as "identical" in tables.

## K. Outstanding
- Not run: latent-dimension ablation (d in {8, 16, 32, 128}), component ablations (single-scale, refinement, decoder width), denoising-training protocol N2.
- Estimated Mac cost (MPS, from measured per-epoch times, excluding cooldowns): baseline 6 min per run, matched 20 min, asymmetric 33 min. A reduced sweep (baseline and asymmetric at d = 8, 32, 128) is about 2 h of compute; N2 additionally needs a Trainer change to feed noisy inputs with clean targets.
- Single seed throughout; no CI.

## L. Acceptance Criteria
| Criterion | Status |
|---|---|
| Noise levels and corruption methods documented | PASS |
| Models evaluated under consistent conditions | PASS |
| Ablation results in tables and plots | NOT YET TESTED (not run) |
| Clean-target and noisy-target experiments distinguished | PARTIAL (N1 only) |

## M. Next
Decide the scope of the training-based ablations (see K) before any run; Phase 7 (latent analysis) not started.
