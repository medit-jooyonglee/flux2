#!/usr/bin/env bash
# LoRA fine-tuning launcher for FLUX.2 [klein] 4B, image-editing (smile design) task.
#
# Trains ONLY LoRA adapters on the transformer (DiT) attention/MLP projections.
# text_encoder and vae are frozen unconditionally by the underlying script
# (train_dreambooth_lora_flux2_klein_img2img.py:1182-1186) -- there is no flag to
# enable training them, which matches the standard/recommended approach for this task.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3 ./launch_lora_train.sh /path/to/dataset_dir
#
# dataset_dir must contain a HF `imagefolder`-style layout with a metadata.jsonl
# providing three columns: an input (condition) image, a target image, and an
# instruction caption. See ../dataset/README.md for how to build this.

set -euo pipefail
cd "$(dirname "$0")"

DATASET_DIR="${1:?Usage: $0 <dataset_dir>}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/smile-design-lora}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-4B}"

source .venv/bin/activate

# do_fp8_training is intentionally NOT passed: FP8 training requires compute
# capability >= 8.9 (Ada/Hopper). These are Ampere (RTX A6000, cc 8.6) GPUs --
# confirmed unsupported earlier when we hit the same wall loading FP8 checkpoints
# for inference (see ../../src/flux2/text_encoder.py's _fp8_supported()).
accelerate launch --config_file accelerate_config_2gpu.yaml \
  external/train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --dataset_name="$DATASET_DIR" \
  --image_column="target_image" \
  --cond_image_column="condition_image" \
  --caption_column="instruction" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --cache_latents \
  --use_8bit_adam \
  --optimizer="adamw" \
  --guidance_scale=1 \
  --rank=16 \
  --lora_alpha=16 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --max_train_steps=2000 \
  --checkpointing_steps=250 \
  --validation_epochs=5 \
  --seed=0 \
  --report_to="tensorboard" \
  "$@"
