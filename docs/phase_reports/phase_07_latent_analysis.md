# Phase 7: Latent-Space Analysis (inference-only)

All results use the three existing seed-0 checkpoints (`best.pt`). No autoencoder was trained, no checkpoint or earlier result was modified, and the official test split was never loaded. Single seed throughout.

## A. Phase Summary
Characterise how the trained latents are used (effective rank, collapse), how much a decoder depends on each principal direction of its latent (post-hoc truncation), and how linearly decodable the CIFAR-10 class is from frozen features (linear probe), against pixel and random-init controls.

## B. Objectives
Latent extraction; PCA; effective-dimensionality and collapse diagnostics; linear probe with controls; figures with captions; state what the analyses can and cannot establish.

## C. Work Completed
- `evaluation/latent_analysis.py`: extraction, covariance spectrum, effective rank, participation ratio, PCA (float64 covariance eigendecomposition), truncation, standardisation, torch-based multinomial logistic-regression probe.
- `scripts/latent_analysis.py`, new plot helpers in `utils/visualization.py`, `tests/test_latent_analysis.py` (9 tests; 121 total).
- Results: `experiments/results/latent_analysis_val.{json,md}`, per-run `latent_analysis_val.json`, six figures `reports/figures/latent_*.png`.
- `scikit-learn` is not installed and not needed; t-SNE was skipped (illustrative only).

## D. Technical Decisions and Protocol
- PCA bases, standardisation statistics and probes are fitted on the 45k **train** split; everything is scored on the 5k **validation** split. Labels are used only to fit and score probes.
- Probe: multinomial logistic regression (L-BFGS, 200 iterations), L2 in {1e-4, 1e-3, 1e-2} chosen on a held-out 10% of the train features, then refit on all train features and scored once on validation.
- Controls: raw pixels (3072-d), raw pixels reduced to PCA-64, and the same architectures with random initialisation (seed 0).
- Truncation: `z_k = mean + (z - mean) V_k V_k^T` with `V` from the TRAIN-latent PCA, passed to `decode`.

## E. Files
Added: `src/.../evaluation/latent_analysis.py`, `scripts/latent_analysis.py`, `tests/test_latent_analysis.py`, results and figures above, this report. Modified: `utils/visualization.py` (new functions only).

## F. Commands
```
python scripts/latent_analysis.py        # about 2 minutes, 2 CPU threads, inference only
```

## G. Verification
- `pytest`: 121 passed; `ruff check` and `ruff format --check` clean.
- Synthetic tests: effective rank of isotropic vs low-rank data; orthonormal, descending PCA basis; exact full-rank truncation; a linear-autoencoder stub whose truncation error falls strictly with k; probe separates Gaussians and is at chance for random labels; standardisation uses train statistics only; the script never references the test split.
- Methodological checks against independent re-computations (SVD instead of `eigh`):
  1. Pixel PCA-64: the basis mean equals the TRAIN mean (not the validation mean); the top-64 subspace matches an independent SVD of train pixels (minimum principal-angle cosine 1.000000); validation is projected with the train mean and basis (validation projections are not re-centred: mean offset up to 0.07); train and validation indices are disjoint.
  2. Truncation: principal components are ordered by TRAIN-latent variance (eigenvalues match an independent SVD); the production truncation curve equals an independent re-computation to 1e-9 (asymmetric) and 3e-10 (baseline) at k = 0, 4, 16, 32, 64; decoding the truncated latent differs from decoding the original (e.g. 0.0263 vs 0.0077 for k = 8), confirming the decoder receives the truncated code; k = 64 reproduces each model's recorded validation MSE.
- Model checkpoints' sha256 hashes were identical before and after the analysis.

## H. Results (seed 0; 64-d latents)

Linear probe, validation accuracy (chance 10%, standard error about 0.7 points):
| Representation | Features | Accuracy | Train acc. |
|---|---|---|---|
| raw pixels | 3072 | 41.2% | 45.9% |
| raw pixels PCA-64 | 64 | 38.0% | 39.2% |
| baseline (trained) | 64 | 37.9% | 38.1% |
| baseline_matched (trained) | 64 | 37.5% | 38.4% |
| asymmetric (trained) | 64 | 37.9% | 38.3% |
| baseline (random init) | 64 | 32.4% | 33.8% |
| baseline_matched (random init) | 64 | 33.4% | 34.7% |
| asymmetric (random init) | 64 | 32.4% | 33.4% |

Latent usage (train split):
| Model | Effective rank | Participation ratio | Dims for 90/95/99% variance | Low-variance dims | Random-init eff. rank |
|---|---|---|---|---|---|
| baseline | 29.3 | 13.5 | 39/49/61 | 0 | 20.2 |
| baseline_matched | 37.2 | 18.4 | 45/54/62 | 0 | 16.0 |
| asymmetric | 31.3 | 14.5 | 41/50/61 | 0 | 14.9 |

Validation MSE after post-hoc truncation to k principal components:
| Model | k=0 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| baseline | 0.0625 | 0.0308 | 0.0236 | 0.0174 | 0.0119 | 0.0092 | 0.00759 |
| baseline_matched | 0.0625 | 0.0309 | 0.0237 | 0.0175 | 0.0118 | 0.0092 | 0.00749 |
| asymmetric | 0.0625 | 0.0317 | 0.0262 | 0.0216 | 0.0163 | 0.0114 | 0.00766 |

Figures: `latent_probe_accuracy.png`, `latent_explained_variance.png`, `latent_truncation_mse.png`, `latent_pca_{baseline,baseline_matched,asymmetric}.png` (validation latents on the first two PCs, coloured by class).

## I. Interpretation and Hypotheses
- **H4 (probe quality): not supported.** The three trained latents are indistinguishable (37.5-37.9%, within about one standard error). Training adds about 4-5 points over random-init encoders, but no trained latent beats a plain PCA-64 of the pixels (38.0%); the 3072-d pixels reach 41.2%. The linear probe underfits rather than overfits (train accuracy close to validation).
- **H5 (collapse): not supported as an advantage; no collapse observed.** No model has low-variance dimensions and each uses roughly half of the 64 dimensions. The asymmetric effective rank (31.3) is slightly above the baseline's (29.3) and below the matched baseline's (37.2). Absolute latent scale (std 5.1 vs 0.4-0.8) is arbitrary and not informative.
- **H2 (compactness): not tested.** Post-hoc truncation is only a proxy: the asymmetric decoder loses more than the baselines at every k < 64 (e.g. 37% higher MSE at k = 32), i.e. its information is spread over more directions. This does not equal training with a smaller latent dimension.
- **PCA scatter plots show no visible class structure**; they are illustrative and support no claim about semantic organisation.

## J. Issues Encountered
- A plotting expression had a misplaced parenthesis and crashed after the numeric results were written; fixed and the analysis re-run in full.
- My first independent truncation check crashed (MPS has no float64); fixed in the check script only, not in the analysed code.
- A first version of one unit test asserted that decoding the mean latent is worse than the full latent for an untrained model; that was a test flaw (an untrained model barely depends on its latent) and was replaced by a stub model where the latent matters.

## K. Limitations / Outstanding
- Single training seed; differences of about 1.5 probe points or less are not interpretable.
- Linear probe and effective-rank measures are proxies; no claim about general semantic quality.
- Probes use 5k validation images; PCA/probe statistics from 45k train images.
- The latent-dimension sweep (H2) was not run (compute and thermal limits).
- t-SNE/UMAP not produced.

## L. Acceptance Criteria
| Criterion | Status |
|---|---|
| Latent extraction works | PASS |
| Visualisations reproducible | PASS (deterministic; `scripts/latent_analysis.py`) |
| Figures have titles, labels, legends, captions | PASS (captions are in this report) |
| Conclusions supported by measurable evidence | PASS (H4/H5 evaluated; H2 explicitly not tested) |

## M. Next
Phase 8 (final report) is not started and awaits approval.
