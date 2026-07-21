#!/usr/bin/env bash
# Crash-resilient PTv3 training: retries after non-zero exits, warm-
# starting ONLY from a checkpoint that passes check_checkpoint.py.
#
# A crash can leave NaN'd weights in model_last.pth (fp16 BN-stat
# overflow was the original source); blindly warm-starting from it
# collapses training and its junk "best" then overwrites the real one.
# So each attempt: verify candidates (best, last, then *_verified
# snapshots), warm from the first clean one, and after the attempt
# snapshot a passing model_best to model_best_verified_<n>.pth so good
# weights are never overwritten by a later bad attempt.
#
# Usage: train_until_done.sh <dataset_dir> <run_name> --python-exe P [args...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="$1"; RUN="$2"; shift 2
MAX_TRIES=8
MODEL_DIR="$DATASET/training_runs/$RUN/model"
mkdir -p "$MODEL_DIR"

# check_checkpoint re-execs under the training python (torch lives
# there); find --python-exe among the passthrough args.
PYEXE=""
prev=""
for a in "$@"; do
    [ "$prev" = "--python-exe" ] && PYEXE="$a"
    prev="$a"
done

verify() { # verify <ckpt> -> 0 clean
    [ -f "$1" ] || return 1
    if [ -n "$PYEXE" ]; then
        "$PYEXE" "$ROOT/tools/check_checkpoint.py" "$1" >/dev/null 2>&1
    else
        "$ROOT/.venv/bin/python" "$ROOT/tools/check_checkpoint.py" "$1" \
            --python-exe "$PYEXE" >/dev/null 2>&1
    fi
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 $MAX_TRIES); do
    WARM=()
    for c in "$MODEL_DIR/model_best.pth" "$MODEL_DIR/model_last.pth" \
             "$MODEL_DIR"/model_best_verified_*.pth \
             "$MODEL_DIR"/model_best_r1_*.pth; do
        if verify "$c"; then
            WARM=(--weight "$c")
            echo "=== warm source (verified): $(basename "$c") ==="
            break
        elif [ -f "$c" ]; then
            echo "=== quarantining unverified $(basename "$c") ==="
            mv "$c" "$c.quarantined"
        fi
    done
    echo "=== attempt $i/$MAX_TRIES $(date '+%H:%M:%S') ${WARM[*]:-cold-start} ==="
    "$ROOT/.venv/bin/python" -u "$ROOT/tools/launch_training.py" \
        --dataset "$DATASET" --run-name "$RUN" "${WARM[@]}" "$@"
    rc=$?
    if verify "$MODEL_DIR/model_best.pth"; then
        cp "$MODEL_DIR/model_best.pth" \
           "$MODEL_DIR/model_best_verified_$i.pth"
        echo "=== snapshot: model_best_verified_$i.pth ==="
    fi
    if [ $rc -eq 0 ]; then
        echo "=== training completed on attempt $i ==="
        exit 0
    fi
    echo "=== attempt $i exited rc=$rc — retrying ==="
    sleep 5
done
echo "=== gave up after $MAX_TRIES attempts ==="
exit 1
