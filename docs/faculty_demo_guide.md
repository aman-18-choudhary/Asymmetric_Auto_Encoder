# Faculty Demonstration Guide

**Project:** Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning
**Primary source for every research claim:** [`reports/final_report.md`](../reports/final_report.md)

## 1. Objective in one paragraph

A symmetric autoencoder can reconstruct well while learning a latent code of little use, because a strong
decoder can compensate for a weak code. We tested whether an **asymmetric** design (a heavy residual encoder, a
tight 64-dimensional bottleneck and a deliberately light multiscale decoder) learns a more compact, useful and
noise-robust latent representation than a conventional symmetric autoencoder under matched training. This was a
**hypothesis, not an established result**; the experiments were designed so it could fail. The demo lets you
inspect the saved evidence and run the trained models live.

## 2. Launch (from the repository root)

```bash
scripts/run_demo.sh
```

Open **http://localhost:8501** (Streamlit's default port; add `--server.port 8502` to the command if it is taken).
The launcher binds the server to `localhost` only and turns off usage statistics. Stop with `Ctrl+C`. First-time setup (once): `pip install -e ".[demo]"` and
`python scripts/build_demo_assets.py` (see Troubleshooting).

**Before the session:** open the app 10 minutes early, click through all four tabs once (the first visit loads the
models), close CPU-heavy applications, and keep the laptop on mains power.

## 3. What is saved, what is live, what was never run

| Kind | Where you see it | What it is |
|---|---|---|
| **Saved research results** | Results tab | Numbers from the three seed-0 training runs, computed earlier on an Apple M2 (MPS). Includes the validation tables, efficiency, robustness curves and latent analysis. The test numbers are the one-time final evaluation. |
| **Live inference** | Reconstruct, Robustness, Latent tabs | The three trained checkpoints run **on CPU, right now**, on 300 validation images. These are single-image illustrations, not research results. |
| **Not conducted** | Mention verbally | Training with more seeds, the latent-dimension sweep, component ablations, denoising training, other datasets. |

## 4. Suggested 5-8 minute run sheet

| Time | Tab | What to do |
|---|---|---|
| 0:00-1:00 | (banner) | State the objective (Section 1). Point to the banner: **all results come from one training run per model (seed 0)**. |
| 1:00-2:30 | Results | Reconstruction table, then the model size and compute table (Section 6). |
| 2:30-4:00 | Reconstruct | Class filter *frog* or *ship*; compare the three reconstructions and the per-image metrics; open the multiscale panel. |
| 4:00-5:30 | Robustness (N1) | Gaussian 0.10, then 0.30; then salt-and-pepper 0.10 (Section 7). |
| 5:30-7:00 | Latent | Truncation slider, 2-D projection, interpolation (Section 8). |
| 7:00-8:00 | (summary) | Findings and limitations (Sections 9-10). End on the limitations. |

## 5. What to show in each tab

**Results.**
- *Reconstruction, validation split.* "Checkpoints were chosen on this split, so it is a development measure."
- Expand *Saved official-test metrics*: "evaluated **once** at the end; the demo never loads the test images."
- *Model size and compute.* This is the slide that shows the cost of the asymmetric design.
- *Noise robustness.* Switch Gaussian / salt-and-pepper and PSNR / SSIM; the "corrupted input" line is
  the noisy image itself.
- Scroll to *Latent analysis* (linear probe, effective rank, post-hoc truncation) and the saved figures.

**Reconstruct.** Pick one class and one sample. Show original plus three reconstructions and the live-CPU metrics
table. Say clearly that a single image can favour any model: on the default airplane the baseline has the lowest MSE, on
the default frog the asymmetric model is marginally lowest; the dataset-level tables on the Results tab are what count.
Open *Asymmetric model: multiscale decoder outputs* to show the 8x8, 16x16 and 32x32 heads that the multiscale
loss supervises.

**Robustness (N1).** Show clean, corrupted, and the three reconstructions, with scores against the **clean** image.
The first table row is the noisy image's own score.

**Latent.** Truncation slider (k = 4, 16, then 64), the 2-D projection with the selected sample circled, and the
interpolation strip (it opens on two different classes: endpoint A, an airplane, and endpoint B, a dog).

## 6. Narration for the three models

- **baseline** - a conventional symmetric convolutional autoencoder: three stride-2 convolutions, a linear layer to
  a 64-d latent, a mirrored decoder. **450,627 parameters.**
- **baseline_matched** - the *same design widened* so its parameter count matches the asymmetric model
  (**2,812,227**, 0.4% fewer). It exists so that any difference cannot simply be "more parameters". It matches
  parameters, **not compute**.
- **asymmetric** - a heavy residual encoder (about 2.77 M parameters) into the same 64-d latent, then a very light
  decoder (about 50 k parameters) with image heads at 8x8, 16x16 and 32x32 trained with a multiscale loss.
  **2,823,625 parameters.**

All three use identical data, split, optimiser, schedule (30 epochs) and latent size.

## 7. Reading the metrics

| Quantity | Meaning | Better |
|---|---|---|
| **MSE** | Mean squared pixel error, pixels in [0, 1]. | Lower |
| **MAE** | Mean absolute pixel error. | Lower |
| **PSNR (dB)** | `10 log10(1 / MSE)` per image, averaged. A rescaling of MSE, not an independent measure. | Higher |
| **SSIM** | Structural similarity, 0 to 1; sensitive to local structure. | Higher |
| **Parameters** | Trainable weights (model size). | - |
| **MACs per image** | Multiply-accumulates for one forward pass (convolutions and linear layers only). | Lower = cheaper |
| **Latency** | Median forward time on the Apple M2 (CPU/MPS), as saved. Hardware-specific. | Lower |

None of these measures semantic quality. Live numbers in the demo are CPU computations; on the full validation set they
agree with the saved MPS results to within about 1e-5 (relative), so single-image numbers are trustworthy but are
illustrations, not results.

## 8. Demonstrating corruption responsibly (protocol N1)

Say this first: *"The models were trained on clean images only. Here we corrupt the input and score the output against
the clean image. No model was trained to denoise."*

1. **Gaussian 0.10, then 0.30.** At sigma = 0.30 the noisy image scores about 11.9 dB (validation average) while all
   three models recover about 19.5-20.0 dB. Explain why: a 64-d bottleneck gives blurry outputs, which suppresses
   noise structurally; this is **not specific to the asymmetric model**.
2. **Salt-and-pepper 0.10.** Same message.
3. The saved result: the asymmetric model's PSNR drops slightly less from clean to sigma = 0.30 (1.75 dB, versus
   2.25 and 2.37 dB) and at p = 0.10 (1.40 dB versus 1.60 and 1.78 dB). This is **directional, one seed, and not
   established**. It also starts lower on clean images, and a model that discards more detail has less to lose.
   Do **not** call it a denoising advantage.

## 9. Explaining the latent tab

- **PCA.** The 64-d latents are rotated onto their principal components. The basis is fitted on **training-set**
  latents only and never on validation data.
- **Truncation (slider k).** Keep the first k components and decode with the *unchanged, trained* decoder. It shows
  how much of the 64-d code the decoder uses. It is a **post-hoc probe, not the same as training a smaller
  latent**. Saved result: the asymmetric decoder loses more than the baselines at every k from 4 to 48
  (about 38% higher MSE at k = 32 than the baseline).
- **2-D projection.** The first two components, coloured by class; the selected image is circled. There is no visible class
  structure. It is illustrative and proves nothing about semantics.
- **Interpolation.** Each frame decodes `(1 - alpha) * z_A + alpha * z_B`; alpha = 0 reproduces A and alpha = 1
  reproduces B. It is an **exploratory visualisation**, not evidence that the latent space is organised
  semantically; the blur is expected.

## 10. Key findings (final report; single seed)

| Question | Outcome |
|---|---|
| **H1** reconstruction no worse | Not supported by this evidence; not statistically rejected. Test MSE +1.1% vs baseline, +2.3% vs matched; SSIM 0.018 lower. |
| **H2** compactness | **Not fully tested** (no latent-dimension sweep). A post-hoc proxy pointed the other way. |
| **H3** robustness | **Directionally supported, not established.** |
| **H4** linear probe | Not supported: 37.9% (asymmetric) vs 37.9% and 37.5%, equal to a PCA-64 of pixels (38.0%). |
| **H5** collapse | No collapse in any model and no asymmetric advantage (effective rank 31.3 vs 29.3 and 37.2). |
| **Cost** | The asymmetric model needs 5.4x the MACs of the matched baseline and 52x the small baseline's. |

Headline: **the asymmetric design did not demonstrate a reconstruction, compactness or probe advantage, and it
costs substantially more compute.** Its robustness behaviour is the only possibly favourable signal and it is not
established.

## 11. Limitations and open questions

- **One training seed** per model: no variability, no significance tests. Differences of ~1-2% in MSE or ~0.4-0.5 dB
  in noisy PSNR are not interpretable.
- **Reduced plan:** 3 of 9 planned runs; the latent-dimension sweep (H2), ablations and denoising training were
  not run (hardware limits; Colab results were lost).
- One dataset (CIFAR-10, 32x32), one latent size (64), Apple M2/MPS only; no CUDA validation.
- The matched baseline is parameter-matched, **not compute-matched**; the asymmetric model may be under-trained at
  30 epochs.
- Probes and truncation are proxies. **Open questions:** does the picture change at other latent sizes? with more
  seeds? with longer training? with denoising training?

## 12. Likely questions and accurate answers

1. **"Is the asymmetric model better?"** Not on this evidence. Reconstruction is equal or slightly worse, the probe
   is equal, and it costs several times more compute. A small, unproven robustness hint is the only favourable signal.
2. **"Why a matched baseline?"** Without it, any win could just mean more parameters. It matches parameters, not
   compute, which we state as a limitation.
3. **"Why is the asymmetric model so expensive?"** Its encoder works at full 32x32 resolution with residual
   blocks: about 280 M MACs per image against 52 M (matched) and 5 M (baseline).
4. **"Did you train it to denoise?"** No. This is test-time corruption with clean targets (N1). Denoising training
   was planned but not run.
5. **"Why are reconstructions blurry?"** A 64-d bottleneck trained with a pixel loss on 32x32 images; all three
   models are blurry.
6. **"Is the robustness result significant?"** We make no significance claim: one seed, no variance estimate.
7. **"Why one seed?"** Compute and thermal limits on a laptop, and a lost cloud run. We say so rather than hide it.
8. **"Did you use the test set?"** Once, for a final evaluation after all decisions were frozen; nothing was tuned on it.
   The demo never loads it.
9. **"What does truncation prove?"** Only how this trained decoder uses its latent. It is not a test of training at a smaller size.
10. **"Does interpolation show semantic structure?"** No; it is exploratory and the decoded frames are blurry.
11. **"Why is the probe only ~38%?"** A linear probe on a pixel-reconstruction latent; plain PCA of the pixels gives the
    same (38.0%), and raw pixels give 41.2%. Random-initialisation encoders get 32-33%.
12. **"What would you do next?"** More seeds, a latent-size sweep for both architectures, a FLOP-matched baseline, longer
    training, and denoising training.

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| Tabs show "Checkpoints not available" | The three `best.pt` files are git-ignored and exist only where they were trained: `experiments/results/{baseline,baseline_matched,asymmetric}_cifar10_seed0/checkpoints/best.pt`. Restore them (or retrain, see the report's Reproducibility section). The **Results tab still works** from the committed JSON. |
| "Validation samples" / "PCA assets" not available | Run `python scripts/build_demo_assets.py` once (about 30 s; needs the checkpoints and the CIFAR-10 *training* batches in `data/cifar-10-batches-py/`). It reads only the five training batches (`data_batch_1`-`5`) and `batches.meta`. Do not use `scripts/download_data.py` if you want to avoid torchvision's integrity check, which also hashes `test_batch`. |
| `streamlit: command not found` / import errors | Activate `.venv` and run `pip install -e ".[demo]"`. |
| Port already in use | `scripts/run_demo.sh --server.port 8502`. |
| Page reachable from other devices, or usage statistics enabled | You launched `streamlit run` directly. Use `scripts/run_demo.sh`, or add `--server.address localhost --browser.gatherUsageStats false`. |
| A public (Streamlit Community Cloud) copy shows only the Results tab | Expected: the checkpoints and `demo/assets/` are git-ignored and are not in the public repository. Use the local demo for the live tabs. |
| Tables empty at first | Wait a few seconds for the first load; refresh the page. |
| Reset the app | Stop with `Ctrl+C` and start again. |
