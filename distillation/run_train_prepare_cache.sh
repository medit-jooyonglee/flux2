#!/usr/bin/env bash
# Real Phase 1 run: caches the FULL dataset (no --limit) and trains on it -- unlike
# run_smoke_test.sh, which deliberately caches only a tiny subset for a quick pipeline
# sanity check. See README.md and ../docs/flux2_distillation_05b_1b.md section 20/32.
#
# epoch length is len(cached dataset) // batch_size -- with the full dataset this is a
# real epoch (thousands of steps), not the 15-step "epoch" you'd get from the 30-sample
# smoke cache. Make sure CACHE_DIR here is NOT smiledesign_cache_smoke.
#
# This machine's GPUs are shared with other jobs -- pick a genuinely free one first:
#   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
#
# Usage:
#   GPU_ID=<0-7> ./run_train_prepare_cache.sh
#   GPU_ID=3 MAX_STEPS=50000 ./run_train_prepare_cache.sh
#   GPU_ID=3 RESUME=auto ./run_train_prepare_cache.sh                     # continue after a crash/restart
#   GPU_ID=3 RESUME=$OUTPUT_DIR/student_step2000.pt ./run_train_prepare_cache.sh

set -euo pipefail
cd "$(dirname "$0")"

# Without this, python fully buffers stdout as soon as it's not a tty (e.g. piped to a
# log file by nohup/&), so nothing shows up in the log until the internal buffer fills
# or the process exits -- looks like it's hung even though it's actually running fine.
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data1/jooyonglee/smiledesign}"
CACHE_DIR="${CACHE_DIR:-/data1/jooyonglee/smiledesign_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/jooyonglee/smiledesign_runs/my_distil}"
LIMIT="${LIMIT:-}"                 # empty = cache every (image, prompt) pair in DATA_DIR
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_EVERY="${SAVE_EVERY:-1000}"   # periodic checkpoints -- this is a long run on a shared/preemptible box
RESUME="${RESUME:-}"               # 'auto', or a path to a student_step<N>.pt / student_final.pt
GPU_ID="${GPU_ID:-}"

if [ -z "$GPU_ID" ]; then
  echo "! GPU_ID not set (0-7) -- this is a shared machine, check for a free GPU first:"
  echo "  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
  exit 1
fi

# Always call prepare_cache.py rather than skipping based on whether sample 000000 alone
# exists -- that coarse check would wrongly declare the cache "done" if a prior build got
# interrupted partway (e.g. a GPU/CUDA drop) or if DATA_DIR has grown since. The per-sample
# skip inside prepare_cache.py itself (latent.pt + embedding.pt already present) is the
# real resume logic and is cheap to re-check even when there's nothing left to do.
limit_args=()
if [ -n "$LIMIT" ]; then
  limit_args=(--limit="$LIMIT")
fi
echo "Building/resuming cache (${LIMIT:-all}) from $DATA_DIR -> $CACHE_DIR ..."
python3 -u prepare_cache.py \
  --data_dir="$DATA_DIR" \
  --output_dir="$CACHE_DIR" \
  --device="cuda:$GPU_ID" \
  "${limit_args[@]}"

resume_args=()
if [ -n "$RESUME" ]; then
  resume_args=(--resume="$RESUME")
fi

python3 -u train_distill.py \
  --gpu="$GPU_ID" \
  --cache_dir="$CACHE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --max_steps="$MAX_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --save_every="$SAVE_EVERY" \
  --log_every=20 \
  --num_validation_images=4 \
  --validation_num_steps=30 \
  --validation_every_epochs=1 \
  "${resume_args[@]}" \
  "$@"

echo ""
echo "Done. Inspect validation images / loss curves with:"
echo "  tensorboard --logdir=$OUTPUT_DIR"
