# FLUX.2 Klein 4B LoRA

`docs/flux2_klein_4b_lora_training_guide.md`의 권장 흐름을 실행 가능한 형태로 묶은 스크립트다.

- 학습: `FLUX.2-klein-base-4B` + Transformer PEFT LoRA
- 배포 추론: `FLUX.2-klein-4B` + LoRA load/fuse/unload + 4 steps
- Text encoder와 VAE는 vendored Diffusers trainer에서 freeze된다.
- checkpoint에는 LoRA뿐 아니라 optimizer/LR scheduler/random state가 저장되므로 resume할 수 있다.

학습 환경은 [training/README.md](../training/README.md)의 별도 가상환경을 사용한다. 모델이 gated인 경우 먼저 `hf auth login`과 모델 사용 동의가 필요하다.

## 데이터

### Text-to-image / style

가이드 형식을 그대로 사용할 수 있다.

```text
my_t2i/
├── 000001.png
├── 000001.txt
├── 000002.png
└── 000002.txt
```

또는 `metadata.jsonl`을 둘 수 있다. 상대 경로는 데이터셋 루트를 기준으로 해석한다.

```json
{"image":"images/000001.png","caption":"TOK. A portrait in a studio."}
{"file_name":"images/000002.png","text":"TOK. A person outdoors."}
```

### Image editing

같은 stem끼리 condition/target/instruction으로 묶인다. 기하학적 correspondence를 유지해야 한다.

```text
my_edit/
├── input/
│   ├── 000001.png
│   └── 000002.png
├── target/
│   ├── 000001.png
│   └── 000002.png
└── caption/
    ├── 000001.txt
    └── 000002.txt
```

`condition/`은 `input/`의 별칭이다. 또는 아래 `metadata.jsonl` 형식도 지원한다.

```json
{"condition_image":"input/000001.png","target_image":"target/000001.png","instruction":"straighten the visible teeth while preserving identity"}
```

## 학습

```bash
# T2I / style
GPU_IDS=0 ./lora/run_train.sh t2i /data/my_t2i

# Image editing, 2 GPU
GPU_IDS=0,1 \
  ./lora/run_train.sh edit /data/my_edit
```

주요 값은 환경 변수로 한 항목씩 바꿀 수 있다.

```bash
MAX_TRAIN_STEPS=1500 \
CHECKPOINTING_STEPS=250 \
LORA_RANK=16 \
LORA_ALPHA=16 \
LEARNING_RATE=5e-5 \
OUTPUT_DIR=./lora/output/exp-r16-lr5e5 \
  ./lora/run_train.sh edit /data/my_edit
```

resume:

```bash
OUTPUT_DIR=./lora/output/exp-r16-lr5e5 \
  ./lora/run_train.sh edit /data/my_edit --resume_from_checkpoint=latest
```

TensorBoard:

```bash
tensorboard --logdir ./lora/output/exp-r16-lr5e5/logs
```

## LoRA fusion 추론

추론 코드는 다음 순서를 강제한다.

```text
load_lora_weights
  -> fuse_lora(components=["transformer"], lora_scale=...)
  -> unload_lora_weights
  -> fused transformer inference
```

Distilled 4B, 4-step T2I:

```bash
LORA_SCALE=0.7 \
GPU_ID=0 \
  ./lora/inference.sh t2i ./lora/output/t2i-lora \
  "TOK. A cinematic portrait" ./output/t2i.png
```

Distilled 4B, 4-step image editing:

```bash
LORA_SCALE=0.7 \
GPU_ID=0 \
  ./lora/inference.sh edit ./lora/output/edit-lora \
  "straighten the visible teeth while preserving the face and identity" \
  ./output/edit.png ./patient.png
```

같은 LoRA를 Base 조건에서도 확인하려면:

```bash
MODEL_NAME=black-forest-labs/FLUX.2-klein-base-4B \
STEPS=50 GUIDANCE_SCALE=4.0 \
  ./lora/inference.sh t2i ./lora/output/t2i-lora \
  "TOK. A cinematic portrait" ./output/base_validation.png
```

`SAVE_FUSED_MODEL=./lora/output/fused-model`을 지정하면 전체 fused pipeline을 저장할 수 있다. 이는 작은 LoRA adapter가 아니라 4B 전체 모델이므로 디스크 사용량이 크다.
