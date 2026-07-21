#!/usr/bin/env bash
# Crash-resilient wrapper around launch_training.py: warm-starts from
# the run's model_last.pth after any non-zero exit, up to MAX_TRIES.
# Pointcept checkpoints every epoch, so a sporadic CUDA assert costs at
# most one epoch of progress per retry.
#
# Usage: train_until_done.sh <dataset_dir> <run_name> [extra launch args...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="$1"; RUN="$2"; shift 2
MAX_TRIES=8
CKPT="$DATASET/training_runs/$RUN/model/model_last.pth"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 $MAX_TRIES); do
    WARM=()
    [ -f "$CKPT" ] && WARM=(--weight "$CKPT")
    echo "=== attempt $i/$MAX_TRIES $(date '+%H:%M:%S') ${WARM[*]:-cold-start} ==="
    "$ROOT/.venv/bin/python" -u "$ROOT/tools/launch_training.py" \
        --dataset "$DATASET" --run-name "$RUN" "${WARM[@]}" "$@"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== training completed on attempt $i ==="
        exit 0
    fi
    echo "=== attempt $i exited rc=$rc — retrying from last checkpoint ==="
    sleep 5
done
echo "=== gave up after $MAX_TRIES attempts ==="
exit 1
