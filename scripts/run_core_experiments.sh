#!/usr/bin/env bash
# Phase 5 core experiments: 3 models x 3 seeds, identical protocol, sequential (clean timings).
# Seed-major order so partial results stay balanced. Finished runs are skipped; interrupted runs
# resume from checkpoints/last.pt. COOLDOWN_SECONDS (default 0) pauses after each trained run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
for seed in 0 1 2; do
  for cfg in baseline baseline_matched asymmetric; do
    name=$($PY - <<PYEOF
from asymmetric_autoencoder.config import load_config
print(load_config("configs/${cfg}.yaml").experiment_name)
PYEOF
)
    run="experiments/results/${name}_seed${seed}"
    if [ -f "${run}/summary.json" ]; then echo "skip ${run} (finished)"; continue; fi
    resume=""
    [ -f "${run}/checkpoints/last.pt" ] && resume="--resume"
    echo "=== ${cfg} seed ${seed} $(date '+%H:%M:%S') ${resume}"
    $PY scripts/train.py --config "configs/${cfg}.yaml" --seed "${seed}" ${resume}
    echo "--- finished ${run} $(date '+%H:%M:%S'); cooldown ${COOLDOWN_SECONDS:-0}s"
    sleep "${COOLDOWN_SECONDS:-0}"
  done
done
echo "ALL DONE $(date '+%H:%M:%S')"
