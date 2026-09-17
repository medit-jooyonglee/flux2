#!/usr/bin/env bash
# 학습된 LoRA를 load -> fuse -> unload한 뒤 fused Transformer로 추론한다.
#
# Text-to-image:
#   GPU_ID=0 ./lora/inference.sh t2i ./lora/output/t2i-lora \
#     "TOK style portrait, soft studio lighting" ./output/t2i.png
#
# Image editing:
#   GPU_ID=1 ./lora/inference.sh edit ./lora/output/edit-lora \
#     "straighten the teeth while preserving identity" ./output/edit.png ./patient.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-${MODE:-}}"
if [[ $# -gt 0 ]]; then shift; fi
LORA_PATH="${1:-${LORA_PATH:-}}"
if [[ $# -gt 0 ]]; then shift; fi
PROMPT="${1:-${PROMPT:-}}"
if [[ $# -gt 0 ]]; then shift; fi
OUTPUT_PATH="${1:-${OUTPUT_PATH:-$REPO_ROOT/output/lora_result.png}}"
if [[ $# -gt 0 ]]; then shift; fi
INPUT_IMAGE="${INPUT_IMAGE:-}"
if [[ "$MODE" == "edit" ]]; then
  INPUT_IMAGE="${1:-$INPUT_IMAGE}"
  if [[ $# -gt 0 ]]; then shift; fi
fi

if [[ "$MODE" != "t2i" && "$MODE" != "edit" ]]; then
  echo "Usage: $0 <t2i|edit> <lora_path> <prompt> [output.png] [input.png]" >&2
  exit 2
fi
if [[ -z "$LORA_PATH" || -z "$PROMPT" ]]; then
  echo "Both LoRA path and prompt are required." >&2
  exit 2
fi
if [[ "$MODE" == "edit" && ( -z "$INPUT_IMAGE" || ! -f "$INPUT_IMAGE" ) ]]; then
  echo "Edit mode requires an existing input image as the fifth argument." >&2
  exit 2
fi

# 배포 기본 모델은 4-step distilled 4B. Base 검증은 MODEL_NAME을 base-4B로 바꾸고 STEPS=50 사용.
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-4B}"
STEPS="${STEPS:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"

# 추론에 사용할 물리 GPU 번호. CUDA_VISIBLE_DEVICES로 한 장만 노출하므로 Python에서는 cuda:0을 사용한다.
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one GPU index, e.g. 0 or 3: $GPU_ID" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# fusion 시 적용되는 강도. 과편집되면 0.3/0.5/0.7/0.85/1.0 sweep을 권장한다.
LORA_SCALE="${LORA_SCALE:-1.0}"
SEED="${SEED:-42}"
NUM_IMAGES="${NUM_IMAGES:-1}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"

# VRAM이 부족하면 1. 전체 pipeline이 들어가면 0이 더 빠르다.
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"

# 특정 파일을 고를 때만 지정. 기본 trainer 출력은 보통 pytorch_lora_weights.safetensors다.
WEIGHT_NAME="${WEIGHT_NAME:-}"

# 설정하면 adapter가 아닌 전체 fused pipeline도 저장한다(용량이 매우 큼).
SAVE_FUSED_MODEL="${SAVE_FUSED_MODEL:-}"

# 현재 셸에서 활성화된 Conda 환경의 Python만 사용한다.
# 다른 venv/Conda 환경으로 자동 fallback하지 않는다.
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "python was not found in PATH. Activate the intended Conda environment first." >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import torch, diffusers, transformers, peft' >/dev/null 2>&1; then
  echo "Required inference packages cannot be imported with active Python: $PYTHON_BIN" >&2
  echo "Run: $PYTHON_BIN -c 'import torch, diffusers, transformers, peft'" >&2
  exit 2
fi

echo "Python:     $PYTHON_BIN"
ARGS=(
  --mode "$MODE"
  --model "$MODEL_NAME"
  --lora "$LORA_PATH"
  --lora-scale "$LORA_SCALE"
  --prompt "$PROMPT"
  --output "$OUTPUT_PATH"
  --steps "$STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --seed "$SEED"
  --num-images "$NUM_IMAGES"
  --dtype "$DTYPE"
  --device "$DEVICE"
)

if [[ "$MODE" == "edit" ]]; then
  ARGS+=(--input-image "$INPUT_IMAGE")
fi
if [[ "$CPU_OFFLOAD" == "1" ]]; then
  ARGS+=(--cpu-offload)
fi
if [[ -n "$WEIGHT_NAME" ]]; then
  ARGS+=(--weight-name "$WEIGHT_NAME")
fi
if [[ -n "$SAVE_FUSED_MODEL" ]]; then
  ARGS+=(--save-fused-model "$SAVE_FUSED_MODEL")
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/inference.py" "${ARGS[@]}" "$@"
