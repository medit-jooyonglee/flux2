#!/usr/bin/env bash
# Capacity-distilled FLUX.2 Klein Base Student inference.
#
# Usage:
#   GPU_ID=2 ./distillation/run_inferene.sh \
#     /data1/jooyonglee/smiledesign_runs/my_distil/student_step7000.pt \
#     "a natural close-up dental photograph" \
#     ./outputs/distill_step7000
#
# The checkpoint argument may also be the run directory; the highest numbered
# student_step<N>.pt is selected automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT="${1:-${CHECKPOINT:-/data1/jooyonglee/smiledesign_runs/my_distil}}"
if [[ $# -gt 0 ]]; then shift; fi
PROMPT="${1:-${PROMPT:-}}"
if [[ $# -gt 0 ]]; then shift; fi
OUTPUT_DIR="${1:-${OUTPUT_DIR:-$REPO_ROOT/outputs/distillation}}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ -z "$PROMPT" ]]; then
  echo "Usage: $0 <checkpoint-or-run-dir> <prompt> [output-dir] [extra Python args...]" >&2
  exit 2
fi

# Physical GPU exposed to this process. Inside Python it is remapped to cuda:0.
GPU_ID="${GPU_ID:-0}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one integer GPU index: $GPU_ID" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Base 4B is neither guidance-distilled nor step-distilled. Keep CFG=4 and 50
# steps for a quality reference; lower values are explicit speed/quality tests.
NUM_IMAGES="${NUM_IMAGES:-1}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-4.0}"
SEED="${SEED:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
DTYPE="${DTYPE:-bf16}"

# Respect only the currently activated Conda environment.
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python not found; activate the intended Conda environment first." >&2
  exit 2
fi

ARGS=(
  --checkpoint "$CHECKPOINT"
  --prompt "$PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --output-dir "$OUTPUT_DIR"
  --num-images "$NUM_IMAGES"
  --width "$WIDTH"
  --height "$HEIGHT"
  --steps "$STEPS"
  --guidance "$GUIDANCE"
  --dtype "$DTYPE"
  --device cuda:0
)
if [[ -n "$SEED" ]]; then
  ARGS+=(--seed "$SEED")
fi

echo "Python:     $PYTHON_BIN"
echo "GPU:        $GPU_ID (cuda:0 inside Python)"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_DIR"
echo "Images:     $NUM_IMAGES (${WIDTH}x${HEIGHT}, ${STEPS} steps, CFG ${GUIDANCE})"

exec "$PYTHON_BIN" "$SCRIPT_DIR/inference.py" "${ARGS[@]}" "$@"
