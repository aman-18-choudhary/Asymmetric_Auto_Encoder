# Moving Phase 5 training to a cloud GPU (Colab)

Protocol, configs and code are frozen at commit `c670b16` (this document is the only later change).
Nothing here modifies the experiment settings: the same `scripts/run_core_experiments.sh` is used;
it skips finished runs and resumes interrupted ones from `checkpoints/last.pt`.

## Saved progress (from the Mac, MPS)

| Run | Status | Epochs | Resume state |
|---|---|---|---|
| baseline seed 0 | complete | 30/30 | not needed (`summary.json` present) |
| baseline_matched seed 0 | interrupted | 28/30 | `last.pt` valid (epoch 28), `best.pt` = epoch 14 |
| all other 7 runs | not started | 0 | start from scratch |

Expected validation MSE of `best.pt` (use as the CUDA load check): baseline seed 0 = 0.007586,
baseline_matched seed 0 = 0.007486 (best epoch 14; later epochs were marginally worse).

## One-time setup (Colab, Runtime > Change runtime type > GPU)

Upload `phase5_progress.tar.gz`, `cifar-10-python.tar.gz` and `SHA256SUMS` (from `cloud_transfer/`)
to Drive folder `MyDrive/aae_transfer`.

```python
from google.colab import drive; drive.mount('/content/drive')
```

```bash
%%bash
set -e
cd /content
git clone -b feature/phase-5-core-experiments https://github.com/aman-18-choudhary/Asymmetric_Auto_Encoder.git
cd Asymmetric_Auto_Encoder
git log --oneline -1                      # must descend from c670b16 (docs-only change on top)
pip install -q -e .                       # keeps Colab's CUDA build of torch
T=/content/drive/MyDrive/aae_transfer
(cd $T && sha256sum -c SHA256SUMS)        # both files must say OK
mkdir -p /content/drive/MyDrive/aae_results experiments data
tar -xzf $T/phase5_progress.tar.gz -C /content/drive/MyDrive/aae_results --strip-components=2
ln -sfn /content/drive/MyDrive/aae_results experiments/results   # results persist on Drive
tar -xzf $T/cifar-10-python.tar.gz -C data                        # no download, test set not touched
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## CUDA checkpoint check (validation split only; do this before training)

```bash
cd /content/Asymmetric_Auto_Encoder
python scripts/evaluate.py experiments/results/baseline_cifar10_seed0 \
  experiments/results/baseline_matched_cifar10_seed0 --split val
```
Expected `mse` within about 1e-5 of 0.007586 and 0.007486. A load error or a clearly different value
means do not resume; stop and investigate.

## Train / resume (re-run the same cell after any disconnect)

```bash
cd /content/Asymmetric_Auto_Encoder
PYTHON=python bash scripts/run_core_experiments.sh 2>&1 | tee -a experiments/results/core_run_cloud.log
```
Order is seed-major: finish matched seed 0 (epochs 29-30), then asymmetric seed 0, then seeds 1 and 2.

## After training

Download `MyDrive/aae_results` back into `experiments/results/` locally. Run once, in this order,
only when all 9 runs have `summary.json`:

```bash
python scripts/evaluate.py experiments/results/*_seed* --split val --efficiency
python scripts/evaluate.py experiments/results/*_seed* --split test --allow-test   # the single final test evaluation
python scripts/aggregate.py
```
Run `--efficiency` for all 9 runs on one machine so latencies are comparable. On CUDA the
aggregate table's "MPS lat." column prints `n/a` (latency is still in `core_comparison.json`).

## Device-specific limitations (read before comparing numbers)

1. **Not bit-exact across devices.** Resuming matched seed 0 on CUDA continues the MPS trajectory
   with different kernels, so the last two epochs are not the ones an uninterrupted MPS run would
   have produced. The Phase 2 CPU-vs-MPS gap was about 0.7% in validation MSE after 2 epochs; the
   CUDA gap is unmeasured.
2. **Timing is device-specific.** `seconds` per epoch in the two Mac runs are MPS; all other runs
   will be CUDA, so training-time comparisons across models mix devices unless those two runs are
   redone. Parameter counts, FLOPs and metrics are unaffected.
3. **RNG on resume.** Checkpoints saved on the Mac contain Python/NumPy/torch-CPU and MPS RNG state;
   the MPS state is skipped on CUDA (verified by simulation) and there is no CUDA state to restore.
   The data order is unaffected (the shuffling generator is re-seeded from `seed + epoch`).
4. **CUDA is not fully deterministic** (some convolution backward kernels); same-seed CUDA reruns
   may differ in low-order digits. `deterministic: false` in all configs.
5. **`environment.json` is overwritten on resume** with the CUDA environment. The original Mac
   record is preserved in `phase5_progress.tar.gz` (and in `experiments/results/` on the Mac).
6. **Checkpoint loading on CUDA is expected, not yet verified**: all stored tensors are CPU tensors,
   loading uses `weights_only=True` on CPU and `load_state_dict` moves weights to the model's device.
   The check above is the verification.

## Optional: single-device comparison

To avoid mixing devices for the two Mac runs, delete them on the cloud before launching (about
8 minutes of GPU time is a guess, not measured):
```bash
rm -rf experiments/results/baseline_cifar10_seed0 experiments/results/baseline_matched_cifar10_seed0
```
Default above is to resume, per the instruction not to restart completed runs unnecessarily.
