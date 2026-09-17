# FLUX.2 Klein 4B LoRA 학습 가이드

> 대상: **FLUX.2 Klein Base 4B → LoRA 학습 → FLUX.2 Klein 4B Distilled 배포**
>
> 목적: Text-to-Image / Style LoRA / Image Editing LoRA를 안정적으로 학습하고, TensorBoard 기반으로 validation image를 지속적으로 확인하는 실전용 가이드

---

## 1. 전체 전략

FLUX.2 Klein 4B 계열에서는 다음 흐름을 기본으로 권장한다.

```text
                 TRAINING

FLUX.2 Klein Base 4B
        │
        ├─ Transformer weight ── freeze
        ├─ Qwen3-4B           ── freeze
        ├─ VAE                ── freeze
        │
        └─ Transformer LoRA   ── TRAIN
                    │
                    ▼
              LoRA adapter
             (.safetensors)


                 VALIDATION

          LoRA checkpoint
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
Base 4B + LoRA      Distilled 4B + LoRA
  ~50 steps              4 steps
       │                    │
       └─────────┬──────────┘
                 ▼
             TensorBoard
                 +
             PNG 저장


                 DEPLOYMENT

FLUX.2 Klein 4B Distilled
        +
      best LoRA
        ↓
     4-step inference
```

핵심은 **Base 모델에서 학습하고, 최종 배포 품질은 Distilled 4B에서도 반드시 확인하는 것**이다.

---

# 2. Base 4B와 Distilled 4B 역할

## FLUX.2 Klein Base 4B

용도:

- Fine-tuning
- LoRA
- 연구 / customization
- 높은 출력 다양성

특징:

```text
step-distilled      X
guidance-distilled  X
default inference   약 50 steps
```

## FLUX.2 Klein 4B

용도:

- 실제 서비스
- 빠른 inference
- interactive editing

특징:

```text
step-distilled      O
guidance-distilled  O
default inference   4 steps
```

따라서 일반적인 LoRA workflow:

```text
Base 4B
  ↓
LoRA training
  ↓
LoRA adapter
  ↓
Distilled 4B에 load
  ↓
4-step inference
```

> 주의: Base에서 잘 동작하는 adapter가 Distilled에서도 동일한 강도로 동작한다고 무조건 가정하지 말 것.
> 최종 checkpoint 선택 시 **Distilled 4B + LoRA validation을 별도로 수행**하는 것이 좋다.

---

# 3. FLUX.2 Klein 4B 구성

개념적으로:

```text
Prompt
  ↓
Qwen3-4B
Text Encoder
  ↓
Text Embedding
  ↓
FLUX.2 Klein Transformer
  ↓
Latent
  ↓
VAE
  ↓
Image
```

LoRA 학습 시:

| Component | 상태 |
|---|---|
| Qwen3-4B Text Encoder | Freeze |
| FLUX Transformer original weights | Freeze |
| Transformer LoRA A/B | **Train** |
| VAE | Freeze |
| Scheduler / Flow matching | 학습 parameter 없음 |

즉 실제 optimizer에는 **LoRA parameter만 들어간다.**

---

# 4. LoRA가 학습하는 것

LoRA는 원래 weight `W`를 직접 변경하지 않고 작은 low-rank update를 학습한다.

```text
W' = W + ΔW

ΔW = B @ A
```

예:

```text
Original Linear layer
        │
        ├──────────── W ──────────── frozen
        │
        └──── A → B ─────────────── train
```

장점:

- full fine-tuning보다 VRAM 사용량 감소
- checkpoint가 작음
- 빠르게 여러 실험 가능
- Base weight를 오염시키지 않음
- adapter 교체 가능

---

# 4.1 PEFT를 사용하는 이유

FLUX.2 Klein 4B에서 LoRA를 직접 구현할 수도 있지만, 실전에서는 **Hugging Face PEFT(Parameter-Efficient Fine-Tuning)** 를 사용하는 것이 가장 편하다.

PEFT가 담당하는 핵심 기능:

```text
FLUX Transformer
      │
      ├─ 기존 weight freeze
      │
      ├─ LoRA adapter 삽입
      │      ├─ lora_A
      │      └─ lora_B
      │
      ├─ trainable parameter 관리
      ├─ adapter save / load
      └─ multiple adapter 관리
```

즉 전체 학습 stack은 다음처럼 이해하면 된다.

```text
PyTorch
  │
  ├─ Diffusers
  │    └─ FLUX.2 model / pipeline / scheduler
  │
  ├─ Transformers
  │    └─ Qwen3-4B text encoder
  │
  ├─ PEFT
  │    └─ LoRA adapter 생성 / 삽입 / 저장 / 로드
  │
  ├─ Accelerate
  │    └─ BF16 / multi-GPU / gradient accumulation
  │
  ├─ Safetensors
  │    └─ LoRA checkpoint 저장
  │
  └─ TensorBoard
       └─ loss / lr / validation image
```

---

# 4.2 필수 학습 라이브러리

기본 설치:

```bash
pip install \
    torch torchvision \
    diffusers transformers \
    peft accelerate \
    safetensors tensorboard \
    datasets
```

선택적으로:

```bash
pip install bitsandbytes
```

용도:

- optimizer / quantization memory 절감
- QLoRA 계열 실험 시 사용 가능

Attention 최적화가 추가로 필요하다면 현재 PyTorch SDPA / Flash Attention 지원 여부를 먼저 확인하고, 필요할 때만 별도 최적화 라이브러리를 추가한다.

> PEFT, Diffusers, Transformers는 서로 API 호환성이 중요하므로 처음 환경을 만든 뒤 버전을 `requirements.txt` 또는 `pip freeze`로 고정하는 것을 권장한다.

---

# 4.3 PEFT `LoraConfig`

기본 형태:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    init_lora_weights="gaussian",
    target_modules=[
        # FLUX.2 Klein Transformer에서 실제 적용할 module 이름
    ],
)

transformer.add_adapter(
    lora_config,
    adapter_name="dental_lora",
)
```

주요 parameter:

| Parameter | 의미 | 시작값 |
|---|---|---:|
| `r` | LoRA rank | 16 |
| `lora_alpha` | LoRA scaling | 16 |
| `lora_dropout` | adapter dropout | 0.0 ~ 0.05 |
| `target_modules` | LoRA 적용 layer | FLUX 구조 확인 필요 |
| `init_lora_weights` | LoRA 초기화 방식 | 기본값 또는 gaussian |

초기 실험은:

```text
r = 16
alpha = 16
```

으로 시작하고 transformation capacity가 부족할 때:

```text
16 → 32
```

를 비교한다.

---

# 4.4 `target_modules`가 가장 중요함

SDXL / 일반 Transformer에서 사용한 module 이름을 FLUX.2 Klein에 그대로 복사하면 안 된다.

예를 들어 SDXL에서 자주 보는:

```python
target_modules=[
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
]
```

같은 설정은 **FLUX.2 Klein의 실제 module tree와 일치하는지 반드시 확인**해야 한다.

먼저 모델에서 Linear layer 이름을 확인한다.

```python
import torch

for name, module in transformer.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

그 후 실제 FLUX.2 Klein 구현 / 검증된 trainer가 사용하는 target mapping을 기준으로 정한다.

PEFT는 다음도 지원한다.

```python
LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
)
```

`all-linear`은 Transformer의 거의 모든 Linear layer에 LoRA를 삽입하는 강한 설정이다.

장점:

- target layer 이름을 일일이 관리할 필요가 적음
- capacity가 큼

단점:

- trainable parameter 증가
- VRAM / optimizer state 증가
- 과적합 가능성 증가
- 특정 FLUX 구조에서 불필요한 layer까지 학습할 수 있음

따라서 **첫 실험에서는 검증된 FLUX.2 Klein target mapping을 우선 사용**하고, `all-linear`은 ablation으로 비교하는 것을 권장한다.

---

# 4.5 LoRA 삽입 후 trainable parameter 확인

LoRA를 붙였다고 바로 학습하지 말고, 실제로 **LoRA만 `requires_grad=True`인지 확인**한다.

```python
trainable = 0
total = 0

for name, p in transformer.named_parameters():
    total += p.numel()

    if p.requires_grad:
        trainable += p.numel()
        print("TRAIN:", name, p.shape)

print(f"trainable: {trainable:,}")
print(f"total:     {total:,}")
print(f"ratio:     {100.0 * trainable / total:.4f}%")
```

정상적인 LoRA training이면:

```text
전체 4B Transformer parameter 중
아주 작은 일부만 trainable
```

이어야 한다.

optimizer 역시 안전하게 trainable parameter만 전달한다.

```python
trainable_params = [
    p for p in transformer.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=1e-4,
)
```

> `optimizer = AdamW(transformer.parameters(), ...)`도 freeze가 정확하다면 동작할 수 있지만, 명시적으로 trainable parameter만 넘기는 편이 실수 방지에 좋다.

---

# 4.6 PEFT adapter 저장

PEFT의 목적은 전체 4B model을 다시 저장하는 것이 아니라 **작은 adapter만 저장**하는 것이다.

모델 수준에서 PEFT API를 직접 사용하는 경우:

```python
transformer.save_pretrained(
    "./output/checkpoint-1000",
    safe_serialization=True,
)
```

환경 / Diffusers integration에 따라 FLUX pipeline용 LoRA 저장 helper를 사용할 수도 있다.

개념적으로 저장되는 것은:

```text
checkpoint-1000/
├── adapter_config.json
└── adapter_model.safetensors
```

또는 Diffusers-compatible LoRA weight 파일이다.

전체 Base 4B weight를 매 checkpoint마다 복사해서 저장할 필요는 없다.

---

# 4.7 Diffusers에서 LoRA 로드

최종 inference에서는 Base / Distilled 모델을 먼저 로드하고 adapter를 별도로 붙인다.

개념 예시:

```python
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.load_lora_weights(
    "./output/best_lora"
)
```

Diffusers에는 Transformer용 LoRA loader가 통합되어 있으므로, 가능한 경우 pipeline의 `load_lora_weights()` 경로를 사용하는 것이 편하다.

LoRA를 잠시 끄고 Base와 비교할 때는 해당 Diffusers / PEFT 버전에서 제공하는 adapter enable/disable API를 사용한다.

```text
Base
vs
Base + LoRA
```

비교는 반드시 동일 seed / prompt로 수행한다.

---

# 4.8 여러 LoRA Adapter 관리

PEFT는 한 모델에 여러 adapter를 붙이는 workflow를 지원한다.

예:

```text
base FLUX.2
 ├─ dental_alignment
 ├─ dental_whitening
 └─ smile_style
```

개념적으로:

```python
transformer.add_adapter(
    alignment_config,
    adapter_name="alignment",
)

transformer.add_adapter(
    whitening_config,
    adapter_name="whitening",
)
```

이후 특정 adapter를 활성화해서 비교할 수 있다.

다만 production에서는 처음부터 여러 LoRA를 섞기보다:

```text
단일 LoRA 품질 검증
→ adapter composition 실험
```

순서가 안전하다.

---

# 4.9 PEFT checkpoint와 Training Resume는 다름

매우 중요한 주의점.

PEFT adapter만 저장하면 일반적으로:

```text
LoRA weight
LoRA config
```

는 저장되지만 전체 training state가 자동으로 모두 보존된다고 가정하면 안 된다.

정확한 resume에는:

```text
LoRA weight
optimizer state
LR scheduler state
global_step
Accelerate state
random state
```

가 필요하다.

따라서:

```text
inference용 checkpoint
≠
training resume checkpoint
```

으로 구분한다.

권장 구조:

```text
output/
├── adapters/
│   ├── step_0250/
│   ├── step_0500/
│   └── best/
│
└── train_state/
    ├── checkpoint-250/
    └── checkpoint-500/
```

Accelerate를 사용한다면 학습 상태 저장 / 복구를 trainer에서 명시적으로 관리한다.

---

# 4.10 PEFT 관련 실수 체크리스트

```text
[ ] peft 설치 / Diffusers와 버전 호환 확인
[ ] Base FLUX Transformer가 freeze인지 확인
[ ] Qwen3-4B freeze 확인
[ ] VAE freeze 확인
[ ] target_modules가 실제 FLUX.2 module 이름과 일치
[ ] LoRA A/B만 requires_grad=True
[ ] trainable parameter 수 출력
[ ] optimizer에 trainable parameter만 전달
[ ] adapter만 별도 저장
[ ] validation에서 adapter load 정상 확인
[ ] inference checkpoint와 resume checkpoint 구분
```

첫 iteration 전에 다음을 반드시 출력해보는 것을 권장한다.

```python
for name, p in transformer.named_parameters():
    if p.requires_grad:
        print(name)
```

여기서 예상치 못한 Base parameter가 보이면 학습을 시작하지 말고 freeze / target mapping부터 수정한다.

---

# 5. Dataset 종류

## 5.1 Style / Text-to-Image LoRA

가장 단순한 dataset:

```text
image
+
caption
```

예:

```text
dataset/
├── 000001.png
├── 000001.txt
├── 000002.png
├── 000002.txt
└── ...
```

caption:

```text
AVNGRS1. A futuristic hero wearing metallic armor, standing in a city.
```

### Style LoRA caption 원칙

스타일 자체를 배우게 할 경우:

```text
좋음:
AVNGRS1. A man wearing futuristic armor standing in a city.

주의:
AVNGRS1. A comic-book style futuristic superhero...
```

스타일을 caption 안에 직접 반복해서 적으면 모델이:

```text
trigger → style
```

대신

```text
"comic-book style" → style
```

로 의존할 수 있다.

따라서 **항상 유지하고 싶은 스타일 속성은 caption에서 빼고**, 나중에 조절하고 싶은 속성만 caption에 명시하는 전략이 좋다.

---

# 6. Image Editing LoRA Dataset

Dental / Smile application에서는 일반 T2I보다 이 구조가 중요하다.

```text
Input Image
+
Target Image
+
Instruction
```

예:

```text
input/
  000001.png

target/
  000001.png

caption/
  000001.txt
```

caption:

```text
straighten the teeth while preserving the face, lips, lighting, and identity
```

전체:

```text
Patient Image
      │
      ├── instruction
      │
      ▼
FLUX + LoRA
      │
      ▼
Target Smile
```

추천 edit 종류:

```text
alignment
whitening
tooth shape
tooth length
spacing
rotation correction
smile width
reference tooth replacement
identity preservation
no-op
```

---

# 7. Dental Edit Dataset 추천 구성

예:

```text
40%  tooth alignment / geometry edit
20%  tooth shade / whitening
15%  reference tooth replacement
10%  smile adjustment
10%  identity-preserving no-op
 5%  difficult / edge cases
```

No-op 데이터:

```text
Input  = Original
Target = Original

Prompt:
"preserve the image without changing the face or dental appearance"
```

또는 아주 미세한 edit:

```text
"preserve the image and only refine tiny dental details"
```

No-op은 얼굴이나 배경이 불필요하게 바뀌는 문제를 줄이는 데 도움이 된다.

---

# 8. Edit Prompt 다양성

매우 중요하다.

다음 한 문장만 100% 반복:

```text
straighten the teeth
```

하면 해당 wording에는 강하지만 다른 표현에 약해질 수 있다.

따라서 동일 task라도 약 5~10개 표현을 섞는 것이 좋다.

예:

```text
straighten the visible teeth naturally

correct the alignment of the visible teeth

make the front teeth more evenly aligned

improve tooth alignment while preserving the face

adjust the visible teeth into a natural aligned arrangement

correct minor tooth rotation while keeping facial identity
```

---

# 9. Offline Cache 전략

LoRA training에서 가장 낭비가 큰 부분:

```text
Prompt → Qwen3-4B
Image  → VAE
```

이 둘은 대부분 freeze 상태이므로 매 iteration 반복할 이유가 없다.

권장:

```text
OFFLINE PREPROCESSING

Prompt
 ↓
Qwen3-4B
 ↓
Text Embedding
 ↓
CACHE


Image
 ↓
VAE
 ↓
Latent
 ↓
CACHE
```

학습에서는:

```text
cached text embedding
+
cached latent
+
random noise
+
random timestep
        ↓
FLUX Transformer + LoRA
```

만 실행한다.

---

# 10. Cache 권장 항목

예:

```text
cache/
├── 000001_text.safetensors
├── 000001_target_latent.safetensors
├── 000001_input_latent.safetensors
├── 000002_text.safetensors
└── ...
```

또는 shard:

```text
cache_0000.safetensors
cache_0001.safetensors
...
```

큰 데이터에서는 shard 방식이 I/O에 유리할 수 있다.

---

# 11. Text Embedding Cache 주의

FLUX.2의 text conditioning 결과를 직접 cache할 때는 **공식 pipeline / text encoder가 반환하는 embedding 구조를 그대로 저장**하는 것이 안전하다.

단순하게:

```python
qwen(...).last_hidden_state
```

만 저장하는 방식은 모델 구현이 실제로 사용하는 conditioning과 다를 수 있다.

권장:

```text
FLUX.2 official encode prompt
        ↓
실제 transformer 입력 직전 conditioning
        ↓
cache
```

---

# 12. Latent Cache와 Augmentation 충돌

중요한 주의점.

다음 augmentation을 training 중 매번 적용하고 싶다면:

```text
random crop
random resize
random flip
color augmentation
```

latent를 미리 cache하면 augmentation이 고정된다.

즉:

```text
Image
 ↓ augmentation
 ↓ VAE
 ↓ cached latent
```

이 되어 이후 augmentation variation이 사라진다.

선택:

### 방법 A

augmentation 먼저 여러 variant 생성:

```text
image
 ↓
augmentation variants
 ↓
VAE
 ↓
multiple latent cache
```

### 방법 B

VAE online:

```text
image
 ↓ random augmentation
 ↓ VAE
 ↓ training
```

Dental editing은 pixel/geometry correspondence가 중요하기 때문에 **과도한 augmentation은 오히려 위험**하다.

특히:

```text
random asymmetric crop
large rotation
random perspective
horizontal flip
```

은 치아 위치/좌우 의미를 훼손할 수 있다.

---

# 13. Training Flow

일반적인 Flow Matching LoRA 학습:

```text
Clean latent x0
        │
Noise ε │
        │
Random timestep t
        ▼
       x_t
        │
        ▼
FLUX Transformer
   + LoRA
        │
        ▼
predicted flow
        │
        ▼
target flow와 loss
        │
        ▼
backprop
        │
        ▼
LoRA A/B update
```

Base Transformer 자체는 update하지 않는다.

---

# 14. 권장 초기 설정 — A6000 48GB

첫 baseline:

```yaml
model:
  name: black-forest-labs/FLUX.2-klein-base-4B

training:
  resolution: 1024

  mixed_precision: bf16

  batch_size: 1
  gradient_accumulation_steps: 2

  learning_rate: 1.0e-4

  lora_rank: 16
  lora_alpha: 16

  optimizer: adamw

  max_train_steps: 2000

  checkpoint_every: 250
  validation_every: 250
```

A6000에서는 BF16 사용을 우선 권장한다.

---

# 15. Rank 선택

초기:

```text
rank 8
```

매우 가벼운 style test.

일반적인 시작점:

```text
rank 16
```

Dental / Edit behavior처럼 복잡한 transformation:

```text
rank 16
→ 부족하면
rank 32
```

권장.

처음부터 rank를 지나치게 높게 잡으면:

- VRAM 증가
- training parameter 증가
- overfit 가능성 증가
- adapter 용량 증가

가 발생한다.

---

# 16. Learning Rate

좋은 baseline:

```text
1e-4
```

불안정하거나 overfit이 빠르면:

```text
5e-5
```

강하게 학습이 안 되면 데이터/LoRA rank를 먼저 점검하고, 무조건 LR부터 크게 올리지는 않는다.

추천 비교:

```text
1e-4
5e-5
```

---

# 17. Training Step

Style LoRA에서는 데이터가 작으면 수백~수천 step만으로도 충분할 수 있다.

추천 first experiment:

```text
checkpoint:
250
500
750
1000
1250
1500
1750
2000
```

**final checkpoint가 best checkpoint라는 보장은 없다.**

Loss:

```text
계속 감소
```

하지만 visual quality:

```text
좋아짐
→ peak
→ overfit
```

이 될 수 있다.

따라서 checkpoint는 **validation image를 보고 선택**한다.

---

# 18. Overfitting 증상

다음 증상이 나오면 의심:

- 모든 output의 얼굴이 training sample과 비슷해짐
- 배경까지 training image를 복제
- prompt 변경이 잘 안 먹음
- trigger가 너무 강함
- 색감/조명이 하나로 고정
- 치아 edit하면서 얼굴 전체가 변형
- artifact 증가
- LoRA scale 1.0에서 이미지 붕괴

대응:

```text
step 감소
LR 감소
dataset diversity 증가
caption variation 증가
rank 감소
no-op sample 증가
```

---

# 19. TensorBoard를 Main Monitor로 사용

추천 log:

```text
train/loss
train/lr
train/grad_norm

validation/base_50step
validation/distilled_4step
```

가능하면 validation image를 직접 TensorBoard에 넣는다.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./output/tensorboard")
```

metric:

```python
writer.add_scalar(
    "train/loss",
    loss.item(),
    global_step,
)
```

image:

```python
writer.add_image(
    "validation/base_50step",
    image_grid,
    global_step,
)

writer.add_image(
    "validation/distilled_4step",
    distilled_grid,
    global_step,
)
```

---

# 20. Validation은 반드시 고정

항상 동일한:

```text
prompt
seed
input image
resolution
guidance
inference step
LoRA scale
```

를 사용한다.

예:

```text
seed = 42
```

step별:

```text
checkpoint-250
checkpoint-500
checkpoint-750
checkpoint-1000
...
```

만 바꿔서 비교한다.

---

# 21. Base Validation

예:

```text
FLUX.2 Klein Base 4B
+
current LoRA
+
50 steps
```

목적:

- 실제 LoRA가 Base training distribution에서 제대로 학습됐는지 확인
- 학습 자체 문제와 distilled compatibility 문제를 분리

---

# 22. Distilled Validation

최종 서비스 조건:

```text
FLUX.2 Klein 4B
+
same LoRA
+
4 steps
```

목적:

- 실제 배포 조건의 품질 확인
- Base에서는 좋은데 Distilled에서 약해지는 checkpoint 발견
- LoRA scale 재조정

추천:

```text
Base validation:
매 250 step

Distilled validation:
매 500 step
```

---

# 23. Validation Prompt 예시 — Dental

## Alignment

```text
straighten the visible teeth naturally while preserving the face, lips, gums, skin, lighting, and identity
```

## Whitening

```text
make the visible teeth slightly brighter and cleaner while keeping the result realistic and preserving the face
```

## Smile

```text
make the smile slightly wider and more natural while preserving identity and facial structure
```

## No-op

```text
preserve the image, identity, lighting, face, lips, and teeth with minimal changes
```

## Reference tooth

```text
replace the visible tooth design using the provided reference while preserving the face, lips, gums, lighting, and identity
```

---

# 24. TensorBoard Validation Grid 추천

한 checkpoint에서:

```text
ROW 1
alignment

ROW 2
whitening

ROW 3
reference tooth

ROW 4
no-op

COL 1
Input

COL 2
Base + LoRA

COL 3
Distilled + LoRA
```

이렇게 하면 checkpoint 간 변화가 매우 쉽게 보인다.

---

# 25. PNG도 별도로 저장

TensorBoard만 믿지 말고:

```text
validation/
├── step_000250/
├── step_000500/
├── step_000750/
└── ...
```

형태로 저장하는 것을 권장한다.

예:

```text
step_000500/
├── alignment_seed42.png
├── whitening_seed42.png
├── reference_seed42.png
└── noop_seed42.png
```

---

# 26. LoRA Scale Validation

학습된 LoRA가 강해 보이면 inference strength도 sweep한다.

예:

```text
0.3
0.5
0.7
0.85
1.0
```

결과:

```text
0.5 좋음
1.0 과편집
```

이면 training 실패가 아니라 **adapter strength가 과한 것**일 수도 있다.

---

# 27. Edit LoRA에서 특히 볼 것

Dental application에서는 단순 visual quality보다 preservation을 같이 봐야 한다.

체크:

```text
[ ] 치아만 의도대로 변하는가
[ ] 얼굴 identity 유지
[ ] 눈/코/피부가 바뀌지 않는가
[ ] 입술 형태 유지
[ ] 잇몸 경계 자연스러운가
[ ] 치아 개수가 변하지 않는가
[ ] 배경 유지
[ ] lighting 유지
[ ] 좌우 비대칭 artifact 없는가
[ ] no-op에서 거의 동일하게 유지되는가
```

---

# 28. 학습용 Target 품질

Edit LoRA에서는 target quality가 매우 중요하다.

나쁜 target:

```text
Input 얼굴
→ Target 얼굴까지 달라짐
```

이면 모델은:

```text
teeth edit
+
face edit
```

를 동시에 학습한다.

따라서 이상적인 pair:

```text
Input / Target

face            동일
skin            동일
eyes            동일
hair            동일
background      동일
lighting        동일

teeth           변경
mouth local     필요한 범위만 변경
```

즉 **pixel alignment가 높을수록 좋다.**

---

# 29. Synthetic Dental Data 전략

좋은 구조:

```text
Real Good Smile
       ↓
Synthetic dental degradation
       ↓
Synthetic Bad Smile = Input

Original Real Good = Target
```

예:

```text
rotation
spacing
crowding
length variation
shade variation
asymmetry
missing tooth
midline shift
```

이 방식의 장점:

```text
Input/Target identity 동일
lighting 동일
background 동일
pose 동일
```

---

# 30. Dataset Size

## Style LoRA

빠른 feasibility:

```text
15~40 images
```

정도로도 시도 가능.

## Dental Edit LoRA

초기 test:

```text
100~500 paired samples
```

조금 더 안정적인 실험:

```text
1K~5K pairs
```

실서비스 domain generalization:

```text
수천~수만 pair
```

를 목표로 보는 편이 현실적이다.

수보다 **pair 품질과 diversity**가 중요하다.

---

# 31. Resolution 주의

1024×1024는 안전한 baseline.

다만 source가 실제로:

```text
512×512
```

인데 단순 upscale해서 1024로 학습한다고 새로운 detail 정보가 생기는 것은 아니다.

Dental ROI task라면:

```text
full face 1024
```

또는

```text
mouth ROI 512~1024
```

를 별도 실험하는 것이 효율적일 수 있다.

---

# 32. BF16 vs FP16

A6000 48GB:

```text
BF16 우선
```

추천.

장점:

- FP16보다 dynamic range 안정적
- NaN/overflow 위험 감소

가능하면:

```text
mixed_precision = bf16
```

을 첫 baseline으로 사용.

---

# 33. Gradient Checkpointing

VRAM이 충분하다면 꺼서 속도를 확보할 수 있다.

VRAM 부족 시:

```text
gradient_checkpointing = True
```

사용.

Trade-off:

```text
VRAM ↓
compute ↑
training time ↑
```

---

# 34. Text Encoder는 기본적으로 Freeze

처음에는:

```text
Qwen3-4B freeze
```

를 유지한다.

Text encoder까지 LoRA / fine-tuning 하면:

- VRAM 증가
- 학습 난이도 증가
- prompt semantics 망가질 위험
- cache 전략 사용 어려움

따라서 **먼저 Transformer LoRA만으로 한계를 확인**하는 것이 좋다.

---

# 35. VAE는 Freeze

VAE까지 같이 학습하는 것은 처음에는 권장하지 않는다.

VAE 변경은:

- latent distribution 변화
- reconstruction 성능 변화
- 기존 Transformer와 coupling 변화

를 만들기 때문에 LoRA 문제와 분리하기 어렵다.

```text
VAE freeze
```

로 시작한다.

---

# 36. Checkpoint / Resume

반드시 optimizer state와 global step까지 저장한다.

```text
checkpoint-250
checkpoint-500
...
```

resume 시:

```text
global_step
optimizer
lr scheduler
LoRA state
random state
```

를 복구하는 것이 좋다.

단순 LoRA `.safetensors`만 불러와 계속 학습하면 optimizer state는 초기화된다.

---

# 37. Reproducibility

validation에는:

```python
seed = 42
```

처럼 고정 seed.

training은 필요하면:

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
random.seed(seed)
```

사용.

validation에서 generator를 매번 새로 생성:

```python
generator = torch.Generator("cuda").manual_seed(42)
```

해야 동일 noise로 비교 가능하다.

---

# 38. 첫 실험 추천

## Experiment A

```text
Dataset:
100~500 paired dental samples

Model:
FLUX.2 Klein Base 4B

LoRA:
rank = 16
alpha = 16

Precision:
BF16

LR:
1e-4

Resolution:
1024

Steps:
1500

Checkpoint:
250

Validation:
250

Distilled validation:
500
```

---

# 39. 두 번째 실험

A가 overfit:

```text
lr:
1e-4 → 5e-5
```

A가 transformation 부족:

```text
rank:
16 → 32
```

둘을 동시에 바꾸지 말고 **한 변수씩 변경**한다.

---

# 40. 추천 Ablation

```text
EXP 1
rank16 / lr1e-4

EXP 2
rank16 / lr5e-5

EXP 3
rank32 / lr1e-4

EXP 4
rank32 / lr5e-5
```

고정:

```text
dataset
seed
validation prompts
resolution
steps
```

---

# 41. 결과 비교 Metric

Dental editing에서는 자동 metric만으로 부족하다.

권장:

### Pixel / preservation

```text
LPIPS outside mouth ↓
SSIM outside mouth ↑
Face embedding similarity ↑
```

### Local dental region

```text
teeth segmentation consistency
tooth shape accuracy
reference similarity
```

### Human visual QC

```text
naturalness
identity preservation
lip/gum boundary
artifact
clinical usefulness
```

---

# 42. MICE / Mask 기반 Editing과 LoRA 결합

LoRA와 MICE는 경쟁 관계가 아니다.

```text
LoRA
→ 무엇을 어떻게 편집할지 학습

MICE / mask attention control
→ 어디를 편집할지 공간적으로 제한
```

따라서:

```text
FLUX.2 Klein 4B
+
Dental LoRA
+
Mouth / Teeth Mask
+
MICE-style attention control
```

조합이 가능하다.

목표:

```text
Dental behavior 학습
+
attribute leakage 감소
+
얼굴 보존
```

---

# 43. Production Inference 예시

```python
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.load_lora_weights(
    "./output/best_lora"
)

generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    prompt="a realistic portrait with a natural smile",
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=generator,
).images[0]

image.save("result.png")
```

---

# 44. Edit Inference 개념

```python
reference = Image.open("patient.png").convert("RGB")

image = pipe(
    prompt=(
        "straighten the visible teeth naturally while preserving "
        "the face, lips, skin, lighting, and identity"
    ),
    image=reference,
    num_inference_steps=4,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
```

실제 parameter 이름은 사용 중인 Diffusers / FLUX.2 버전에 맞춰 확인한다.

---

# 45. 가장 흔한 실패 원인

## 1. 너무 오래 학습

```text
loss는 좋음
image는 나쁨
```

→ validation image 기준 checkpoint 선택.

## 2. Dataset diversity 부족

```text
같은 얼굴
같은 배경
같은 포즈
```

→ LoRA가 style/task보다 dataset identity를 암기.

## 3. Caption variation 부족

→ 특정 prompt 문장에만 반응.

## 4. Target에서 얼굴까지 변경

→ dental edit와 identity change를 같이 학습.

## 5. LoRA strength 과도

→ inference scale sweep.

## 6. Cache와 augmentation 충돌

→ cached latent는 augmentation이 고정됨.

## 7. Base만 validation

→ production용 Distilled 모델 결과가 다를 수 있음.

## 8. 여러 hyperparameter 동시 변경

→ 무엇 때문에 좋아졌는지 판단 불가.

---

# 46. Recommended Project Structure

```text
flux2_lora/
├── configs/
│   ├── train_style.yaml
│   └── train_dental_edit.yaml
│
├── data/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── cache_latents.py
│
├── validation/
│   ├── prompts.json
│   ├── inputs/
│   └── run_validation.py
│
├── train_lora.py
├── inference_t2i.py
├── inference_edit.py
│
└── output/
    ├── checkpoints/
    ├── tensorboard/
    └── validation/
```

---

# 47. 최종 추천 Workflow

```text
STEP 1
작은 dataset으로 feasibility 확인

      ↓

STEP 2
Base 4B + rank16 LoRA

      ↓

STEP 3
Qwen embedding / VAE latent cache

      ↓

STEP 4
BF16, lr 1e-4

      ↓

STEP 5
250 step마다 checkpoint

      ↓

STEP 6
TensorBoard validation image 저장

      ↓

STEP 7
Base 50-step 확인

      ↓

STEP 8
Distilled 4-step 확인

      ↓

STEP 9
best visual checkpoint 선택

      ↓

STEP 10
LoRA scale sweep

      ↓

STEP 11
Dental dataset 확대

      ↓

STEP 12
MICE / mask 기반 localized editing 결합
```

---

# 48. 가장 중요한 체크리스트

학습 전:

```text
[ ] Base 4B 사용
[ ] Qwen freeze
[ ] VAE freeze
[ ] LoRA target 확인
[ ] dataset pair alignment 확인
[ ] caption variation 확인
[ ] validation set 분리
```

학습 중:

```text
[ ] TensorBoard loss
[ ] learning rate
[ ] grad norm
[ ] fixed-seed validation
[ ] checkpoint별 PNG 저장
[ ] overfit 확인
```

학습 후:

```text
[ ] Base 4B + LoRA
[ ] Distilled 4B + LoRA
[ ] LoRA scale sweep
[ ] no-op test
[ ] face identity preservation
[ ] dental-only edit 확인
[ ] artifact 확인
```

---

# 49. 참고 자료

- Black Forest Labs FLUX.2 repository  
  https://github.com/black-forest-labs/flux2

- FLUX.2 Klein Base 4B  
  https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B

- Hugging Face / BFL Klein LoRA guide  
  https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora

- Hugging Face Diffusers FLUX.2 guide  
  https://huggingface.co/blog/flux-2


- Hugging Face PEFT LoRA reference  
  https://huggingface.co/docs/peft/main/package_reference/lora

- Hugging Face Transformers PEFT integration  
  https://huggingface.co/docs/transformers/peft

- Hugging Face Diffusers LoRA loader API  
  https://huggingface.co/docs/diffusers/api/loaders/lora

---

## 요약

처음부터 복잡하게 가지 말고 다음 baseline을 먼저 성공시키는 것이 좋다.

```text
FLUX.2 Klein Base 4B
+
Qwen / VAE Freeze
+
rank16 LoRA
+
BF16
+
LR 1e-4
+
1024
+
1.5K~2K steps
+
250-step checkpoint
+
TensorBoard fixed-seed validation
+
Distilled 4B 4-step 최종 확인
```

Dental editing에서는 특히 **training loss보다 validation image / identity preservation / local edit 정확도**를 우선해서 checkpoint를 선택한다.
