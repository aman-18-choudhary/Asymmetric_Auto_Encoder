# Phase 3: Proposed Asymmetric Architecture

## A. Phase Summary
Implement Model B (asymmetric autoencoder) with a constrained residual encoder, latent bottleneck, lightweight multiscale decoder and multiscale objective, behind the same interface as the baseline. No comparison experiments were run.

## B. Objectives
Encoder, bottleneck, decoder, multiscale loss, configurable architecture, documented decisions, parameter comparison, tests, small training run.

## C. Work Completed
- `models/blocks.py`: `conv_bn_act`, `ResBlock`, `UpBlock` (bilinear x2 + conv).
- `models/asymmetric_autoencoder.py`: `AsymmetricAutoencoder`.
- `models/losses.py`: `downsample_target`, `multiscale_mse`, `model_loss`.
- `models/summary.py`: `count_parameters`, `parameter_breakdown` (shared by both models).
- `models/factory.py`: `build_model(ModelConfig, in_channels, image_size)`.
- Config: `model.type` plus asymmetric options; `configs/asymmetric.yaml`.
- `training/simple.py` now optimizes `model_loss` (baseline: plain MSE, unchanged).
- `scripts/train_baseline.py` renamed to `scripts/train_debug.py` and made model-agnostic.
- 23 new tests (74 total).

## D. Architecture and Technical Decisions
Concrete shapes (CIFAR-10, d configurable, default 64):

| Stage | Op | Output |
|---|---|---|
| Stem | Conv3x3 3->64, BN, GELU | 64x32x32 |
| Enc 1 | ResBlock(64), Conv s2 64->128 | 128x16x16 |
| Enc 2 | ResBlock(128), Conv s2 128->256 | 256x8x8 |
| Enc 3 | ResBlock(256), Conv s2 256->256 | 256x4x4 |
| Refine (optional) | one ResBlock(256), same weights applied `refine_steps` times | 256x4x4 |
| Bottleneck | Flatten -> Linear 4096->d (optional LayerNorm) | d |
| Dec seed | Linear d->32*4*4, GELU | 32x4x4 |
| Up 1 | bilinear x2, Conv 32->32 ; head 1x1 -> sigmoid | 8x8 (aux output) |
| Up 2 | bilinear x2, Conv 32->16 ; head | 16x16 (aux output) |
| Up 3 | bilinear x2, Conv 16->16 ; head | 32x32 (final output) |

Decisions:
- Loss = sum_s w_s * MSE(x_hat_s, area_avg(x, size_s)), default w = (0.25, 0.5, 1.0) lowest -> highest resolution. Downsampling uses area averaging and requires integer ratios (error otherwise), so every scale's target is a mathematically consistent coarse version of the input.
- Single-scale ablation: `multiscale: false` removes the auxiliary heads (weights `(1.0,)`).
- "Recursive" = one weight-tied ResBlock applied k times; adds compute, not parameters (tested). Off by default and unproven.
- BatchNorm + GELU as in the blueprint. Upsample+conv instead of transposed conv in the decoder.
- Interface parity with the baseline: `forward`, `encode`, `decode`, `latent_dim`, `in_channels`, `scale_weights`; Model B adds `forward_multiscale`.
- `evaluate_mse` always scores the final-resolution output against the full-resolution input, so reported validation MSE is comparable across models. Training-loss histories are NOT comparable (different objectives).
- Phase 0 hypotheses/protocols are unchanged.

## E. Files Added or Modified
Added: `models/{blocks,asymmetric_autoencoder,losses,summary,factory}.py`, `configs/asymmetric.yaml`, `tests/test_asymmetric_model.py`, this report. Modified: `config.py`, `models/baseline_autoencoder.py` (count_parameters moved, re-exported; `scale_weights=None`), `training/simple.py`, `README.md`. Renamed: `scripts/train_baseline.py` -> `scripts/train_debug.py`.

## F. Commands Executed
`pytest -q`; `ruff check .`; `ruff format .`; `python scripts/train_debug.py --config configs/asymmetric.yaml --epochs 2 --max-batches 60` (MPS and CPU); parameter counts and a real-CIFAR-10 16-image overfit probe via ad-hoc Python.

## G. Verification Results
- `pytest`: 74 passed. `ruff check` clean.
- Shapes: outputs 8x8, 16x16, 32x32 for 1- and 3-channel inputs; latent (B, d); all outputs in [0,1].
- Gradients: every parameter (incl. auxiliary heads and refinement block) receives a finite gradient.
- Loss: unit-tested downsampling (hand-computed 2x2 block means), zero loss for perfect predictions, weighted sum, mismatch errors.
- Small training run (CIFAR-10, 2 epochs x 60 batches, seed 0): completes on MPS and CPU; same-device reruns identical on CPU. Single seed, tiny budget.

Parameter comparison (d=64, 3 channels; measured):

| | Total | Encoder (conv + to_latent) | Decoder |
|---|---|---|---|
| Baseline (Model A) | 450,627 | 224,384 | 226,243 |
| Asymmetric (Model B) | 2,823,625 | 2,773,888 | 49,737 |

So B has 6.3x the total parameters, 12x the encoder and 0.22x the decoder. Total size differs: a parameter-matched baseline is required in Phase 5 (baseline `channels: [64,128,256]` gives 1.27M, `[128,256,512]` gives 4.01M parameters; neither is yet matched to 2.82M). Latent sweep (asym / baseline): d=8 2.57M/0.22M, d=32 2.68M/0.32M, d=128 3.12M/0.71M.

## H. Experimental Results
None claimed. Observations from debugging probes (single seed, not experiments):
- 16 real CIFAR-10 images, 400 Adam steps, final-resolution MSE: baseline 0.0003 vs asymmetric 0.0036 (lr 1e-3) / 0.0032 (lr 2e-3). Both reduce loss steadily; the asymmetric model memorises less quickly, consistent with its 4.5x smaller decoder. This measures memorisation speed, not generalisation or representation quality.
- On i.i.d. noise batches the asymmetric model memorises much more slowly than the baseline (0.025-0.05 after 500 steps vs 0.0001). The unit test therefore uses smooth structured images; the noise result is a property of the design, not evidence of a bug (loss decreases monotonically).
- 2-epoch CIFAR-10 debug runs: val MSE 0.0194 (asymmetric) vs 0.0233 (baseline, Phase 2 review run). Not comparable evidence: tiny budget, one seed, different parameter counts.
- Risk for H1: the decoder may be too weak to reconstruct fine detail; this must be measured in Phase 5/6 rather than assumed.

## I. GitHub Activity
Branch `feature/phase-3-asymmetric-model` from `develop` at `a17e866` (Phase 2 merge, verified on remote). Commits are listed in the completion message once created and verified. Not merged.

## J. Issues Encountered
- Initial overfit test on random noise failed (loss 0.113 -> 0.071). Investigated rather than loosening the threshold: loss decreased steadily, real images overfit well, so the test was changed to smooth structured images and the noise behavior documented.

## K. Outstanding Issues
- No parameter-matched baseline yet; no FLOPs or latency measurement.
- BatchNorm behaves differently in train/eval; batch size 1 in training would fail (not guarded).
- Default decoder widths (32/32/16/16) and loss weights are unproven; no ablations run.
- Test suite does not cover MPS/CUDA for the new model (CPU only; MPS verified by script run).
- No CI; no GitHub PRs.

## L. Acceptance Criteria
| Criterion | Status |
|---|---|
| Architecture implemented and tested | PASS |
| Tensor dimensions correct | PASS |
| Gradients flow through the network | PASS |
| Completes a small training run | PASS |
| Baseline and proposed models have consistent interfaces | PASS |
| Documented decisions and parameter-count comparison | PASS |

## M. Next Phase
Phase 4: shared training infrastructure (checkpoints, resume, history, scheduler, config snapshots, early stopping). Not started.
