#!/usr/bin/env bash
# FLUX.2 Klein Base 4B에서 Transformer LoRA만 학습한다.
#
# 사용 예시:
#   GPU_IDS=0 ./lora/run_train.sh t2i /data/my_style
#   GPU_IDS=2,3 ./lora/run_train.sh edit /data/my_edit
#
# 추가 Diffusers trainer 옵션은 데이터셋 경로 뒤에 그대로 전달할 수 있다.
#   ./lora/run_train.sh t2i /data/my_style --resume_from_checkpoint=latest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="${1:-${TASK:-}}"
if [[ $# -gt 0 ]]; then shift; fi
DATASET_DIR="${1:-${DATASET_DIR:-}}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$TASK" != "t2i" && "$TASK" != "edit" ]]; then
  echo "Usage: $0 <t2i|edit> <dataset_dir> [extra trainer args...]" >&2
  exit 2
fi
if [[ -z "$DATASET_DIR" || ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: ${DATASET_DIR:-<empty>}" >&2
  exit 2
fi

# 학습은 반드시 guidance/step distillation 전의 Base 모델에서 수행한다.
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-base-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/lora/output/${TASK}-lora}"

# 학습에 사용할 물리 GPU 번호. 단일 GPU는 "0", 다중 GPU는 "0,1"처럼 지정한다.
# 지정된 GPU는 프로세스 안에서 cuda:0, cuda:1 순서로 다시 매핑된다.
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be a comma-separated list of GPU indices, e.g. 0 or 2,3: $GPU_IDS" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# 총 optimizer update 횟수. 작은 데이터셋은 1,500~2,000부터 시작하고,
# final이 아닌 validation image가 가장 좋은 checkpoint를 선택한다.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"

# adapter 및 optimizer/LR/random state를 저장하는 간격이다. resume도 이 checkpoint를 사용한다.
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-8}"

# LoRA 표현력. baseline은 rank/alpha 16이며 부족할 때 한 번에 한 변수만 32로 비교한다.
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

# 기본 learning rate. 과적합/불안정 시 5e-5를 비교한다.
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"

# GPU 한 장당 batch와 누적 횟수. 유효 batch = GPU 수 * batch * accumulation.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

# 가이드의 A6000 baseline. BF16은 Ampere 이상에서 FP16보다 안정적이다.
RESOLUTION="${RESOLUTION:-1024}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# 기본 process 수는 GPU_IDS에 지정한 GPU 개수다. 필요할 때만 NUM_PROCESSES로 덮어쓴다.
IFS=, read -r -a SELECTED_GPUS <<< "$GPU_IDS"
NUM_PROCESSES="${NUM_PROCESSES:-${#SELECTED_GPUS[@]}}"
SEED="${SEED:-42}"

# 8-bit AdamW로 optimizer VRAM을 줄인다. bitsandbytes가 없는 환경에서는
# USE_8BIT_ADAM=0으로 일반 torch AdamW를 사용할 수 있다.
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"

# VAE latent cache는 메모리를 절약한다. cache 사용 시 매 step image augmentation은 고정된다.
CACHE_LATENTS="${CACHE_LATENTS:-1}"

# activation VRAM을 줄이는 대신 학습 속도가 느려진다. VRAM 여유가 크면 0으로 비교 가능하다.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# validation은 고정 prompt/seed/image를 사용한다. 이 trainer 옵션은 epoch 단위다.
# T2I: VALIDATION_PROMPT만 설정
# Edit: VALIDATION_PROMPT와 VALIDATION_IMAGE를 함께 설정
VALIDATION_PROMPT="${VALIDATION_PROMPT:-}"
VALIDATION_IMAGE="${VALIDATION_IMAGE:-}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-5}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-1}"

# 현재 셸에서 활성화된 Conda 환경의 Python만 사용한다.
# 다른 venv/Conda 환경으로 자동 fallback하지 않는다.
PYTHON_BIN="$(command -v python || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "python was not found in PATH. Activate the intended Conda environment first." >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import torch, diffusers, accelerate, transformers, datasets, peft' >/dev/null 2>&1; then
  echo "Required training packages cannot be imported with active Python: $PYTHON_BIN" >&2
  echo "Run: $PYTHON_BIN -c 'import torch, diffusers, accelerate, transformers, datasets, peft'" >&2
  exit 2
fi
if [[ "$USE_8BIT_ADAM" == "1" ]] && ! "$PYTHON_BIN" -c 'import bitsandbytes' >/dev/null 2>&1; then
  echo "USE_8BIT_ADAM=1 requires bitsandbytes in active Python: $PYTHON_BIN" >&2
  echo "Install bitsandbytes or run with USE_8BIT_ADAM=0." >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

LAUNCH_ARGS=(--num_processes "$NUM_PROCESSES" --mixed_precision "$MIXED_PRECISION")
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

TRAIN_ARGS=(
  --pretrained_model_name_or_path="$MODEL_NAME"
  --output_dir="$OUTPUT_DIR"
  --resolution="$RESOLUTION"
  --center_crop
  --train_batch_size="$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
  --rank="$LORA_RANK"
  --lora_alpha="$LORA_ALPHA"
  --lora_dropout="$LORA_DROPOUT"
  --optimizer=AdamW
  --learning_rate="$LEARNING_RATE"
  --lr_scheduler=constant_with_warmup
  --lr_warmup_steps="$LR_WARMUP_STEPS"
  --max_train_steps="$MAX_TRAIN_STEPS"
  --checkpointing_steps="$CHECKPOINTING_STEPS"
  --checkpoints_total_limit="$CHECKPOINTS_TOTAL_LIMIT"
  --guidance_scale=1.0
  --mixed_precision="$MIXED_PRECISION"
  --seed="$SEED"
  --report_to=tensorboard
  --logging_dir=logs
  --allow_tf32
)

if [[ "$USE_8BIT_ADAM" == "1" ]]; then
  TRAIN_ARGS+=(--use_8bit_adam)
fi
if [[ "$CACHE_LATENTS" == "1" ]]; then
  TRAIN_ARGS+=(--cache_latents)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  TRAIN_ARGS+=(--gradient_checkpointing)
fi
if [[ -n "$VALIDATION_PROMPT" ]]; then
  TRAIN_ARGS+=(
    --validation_prompt="$VALIDATION_PROMPT"
    --validation_epochs="$VALIDATION_EPOCHS"
    --num_validation_images="$NUM_VALIDATION_IMAGES"
  )
  if [[ "$TASK" == "edit" ]]; then
    if [[ -z "$VALIDATION_IMAGE" || ! -f "$VALIDATION_IMAGE" ]]; then
      echo "Edit validation requires a valid VALIDATION_IMAGE." >&2
      exit 2
    fi
    TRAIN_ARGS+=(--validation_image="$VALIDATION_IMAGE")
  fi
fi

echo "Task:       $TASK"
echo "GPU IDs:    $GPU_IDS ($NUM_PROCESSES process(es))"
echo "Python:     $PYTHON_BIN"
echo "Base model: $MODEL_NAME"
echo "Dataset:    $DATASET_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Steps:      $MAX_TRAIN_STEPS (checkpoint every $CHECKPOINTING_STEPS)"

exec "$PYTHON_BIN" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" \
  "$SCRIPT_DIR/train_lora.py" \
  --task "$TASK" \
  --dataset-dir "$DATASET_DIR" \
  -- \
  "${TRAIN_ARGS[@]}" \
  "$@"
