#!/usr/bin/env bash
# Phase 0 smoke test for the 4B -> ~1B capacity distillation (see README.md and
# ../docs/flux2_distillation_05b_1b.md section 20/32). Confirms forward/shape/finite
# loss/backward/checkpoint/validation-logging all work before committing to a real run.
#
# This machine's GPUs are shared with other jobs -- pick a genuinely free one first:
#   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
#
# Usage:
#   GPU_ID=<0-7> ./run_smoke_test.sh
#   GPU_ID=2 DATA_DIR=/data1/jooyonglee/smiledesign ./run_smoke_test.sh

set -euo pipefail
cd "$(dirname "$0")"

# Without this, python fully buffers stdout as soon as it's not a tty (e.g. piped to a
# log file by nohup/&), so nothing shows up in the log until the internal buffer fills
# or the process exits -- looks like it's hung even though it's actually running fine.
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data1/jooyonglee/smiledesign}"
CACHE_DIR="${CACHE_DIR:-/data1/jooyonglee/smiledesign_cache_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/jooyonglee/smiledesign_runs/first_train}"
LIMIT="${LIMIT:-30}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GPU_ID="${GPU_ID:-}"

if [ -z "$GPU_ID" ]; then
  echo "! GPU_ID not set (0-7) -- this is a shared machine, check for a free GPU first:"
  echo "  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
  exit 1
fi

if [ ! -f "$CACHE_DIR/000000/latent.pt" ]; then
  echo "Building cache ($LIMIT samples) from $DATA_DIR -> $CACHE_DIR ..."
  python3 -u prepare_cache.py \
    --data_dir="$DATA_DIR" \
    --output_dir="$CACHE_DIR" \
    --limit="$LIMIT" \
    --device="cuda:$GPU_ID"
else
  echo "Reusing existing cache at $CACHE_DIR"
fi

python3 -u train_distill.py \
  --gpu="$GPU_ID" \
  --cache_dir="$CACHE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --max_steps="$MAX_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --log_every=1 \
  --num_validation_images=2 \
  --validation_num_steps=10 \
  --validation_every_epochs=1 \
  "$@"

echo ""
echo "Done. Inspect validation images / loss curves with:"
echo "  tensorboard --logdir=$OUTPUT_DIR"
