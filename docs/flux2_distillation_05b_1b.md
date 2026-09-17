# FLUX.2 Klein 4B → 0.5B~1.0B Distillation Experiment Plan

## 1. 목표

FLUX.2 Klein 4B 계열을 Teacher로 사용해 dental/smile editing 특화 **0.5B~1.0B Student**를 만드는 실험 계획이다.

최종 목표:

```text
Patient image
+ optional tooth reference
+ cached task/text embedding
        ↓
0.5B~1.0B Dental Student
        ↓
natural smile / tooth edit
```

장기적으로는 Qwen text encoder runtime 제거 또는 embedding cache, single/multi-reference editing 유지, identity preservation, 치아/잇몸/입술 harmonization, 4~8 step inference를 목표로 한다.

## 2. Teacher 선택

공식 FLUX.2 Klein 4B 계열은 역할이 나뉜다.

```text
FLUX.2 Klein 4B Base
- undistilled
- fine-tuning / LoRA용
- 기본 약 50-step
- Apache-2.0

FLUX.2 Klein 4B
- step-distilled
- guidance-distilled
- 기본 4-step
- production용
- Apache-2.0
```

첫 distillation 실험에서는 **4B Base를 Teacher로 권장**한다.

이유:
- distilled 모델보다 training signal이 풍부
- guidance 사용 가능
- domain LoRA를 붙여 high-quality teacher를 만들기 쉬움

최종 dental teacher:

```text
FLUX.2 Klein 4B Base
+
Dental LoRA
        ↓
Dental Teacher
```

## 3. 가장 중요한 전략

다음 두 문제를 한 번에 해결하지 않는다.

```text
Problem A
4B → 0.5~1B
Model capacity reduction

Problem B
50-step → 4-step
Step distillation
```

권장 순서:

```text
Stage 1
4B Base Teacher
      ↓
0.5~1B Student
20~50-step quality 확보

Stage 2
0.5~1B Student
      ↓
8-step
      ↓
4-step
```

즉 **capacity distillation → step distillation** 순서가 좋다.

## 4. Student 목표 규모

먼저:

```text
Student-A ≈ 1.0B
```

를 만들고 성공 후:

```text
Student-B ≈ 0.5B
```

를 시도한다.

0.5B를 처음부터 목표로 하면 capacity gap이 너무 커서 원인 분석이 어려워질 수 있다.

## 5. Student Architecture 축소

FLUX 계열 Transformer interface를 유지하면서 다음을 줄인다.

- Transformer depth
- hidden width
- attention heads
- MLP/FFN dimension

개념적으로:

```text
Teacher 4B
Depth = D
Width = H

Student 1B
Depth ≈ 0.5~0.7 D
Width ≈ 0.6~0.75 H
```

정확한 크기는 실제 `Klein4BParams`를 기준으로 파라미터 수를 계산해 맞춘다.

0.5~1B까지 줄이려면 보통 **depth + width 동시 축소**가 필요하다.

## 6. Student 초기화

### A. Random initialization
가장 단순하지만 많은 데이터와 학습량이 필요하다.

### B. Teacher layer mapping
Teacher block을 간격을 두고 골라 Student에 초기화한다.

```text
Teacher blocks:
0 1 2 3 4 5 6 7 ...

Student:
0   2   4   6 ...
```

hidden width가 같을 때 특히 쉽다.

### C. Width가 다를 때
weight slicing / projection을 사용할 수 있지만 구현 난이도가 높다.

첫 1B 실험은 가능한 한 Teacher와 latent/input-output interface를 동일하게 유지하고, depth 축소부터 시작하는 것이 디버깅에 유리하다.

## 7. 필요한 데이터 종류

Distillation은 항상 사람이 만든 GT만 필요하지 않다.

Teacher supervision을 함께 사용할 수 있다.

### Dataset A — Text-to-Image

```text
Image
+
Caption
```

Dental-specific Student라면 전체의 10~30% 정도만 유지해도 된다.

목적:
- 기본 visual prior 유지
- 얼굴/피부/조명/구도 품질 유지

### Dataset B — Image Edit Pair

가장 중요.

```text
Source image
+
Instruction
+
Target
```

Dental에서는 다음 방식이 가장 좋다.

```text
Real Good Smile
      ↓ synthetic defect
Synthetic Bad Smile

Input = Synthetic Bad
Target = Original Real Good
```

장점:
- 동일 identity
- 동일 pose
- 동일 expression
- 동일 lighting
- 동일 background

### Dataset C — Reference-conditioned Edit

```text
Patient
+
3D Tooth Raster
+
Instruction
+
Target
```

Target은 다음처럼 반자동 구축한다.

```text
Patient
+
3D Tooth Raster
        ↓
rough geometry composite
        ↓
4B Teacher harmonization
        ↓
auto filter
        ↓
human QC
        ↓
Target
```

## 8. Dataset 규모

### Smoke Test

```text
1K~5K samples
```

목적:
- Teacher/Student forward 확인
- shape 확인
- loss finite 확인
- backward/checkpoint 확인

### First Useful Experiment

```text
20K~50K samples
```

1B Student가 Teacher behavior를 따라가는지 확인.

### Serious Dental Student

```text
100K~500K effective samples
```

예:

```text
10K unique faces
× 5 dental variants
× 2 instruction variants
= 100K samples
```

### Generic FLUX capability까지 넓게 보존

수백만~수천만 sample이 필요할 수 있다.

따라서 0.5B~1B 목표에서는 **범용 FLUX clone보다 dental-specific Student가 현실적**이다.

## 9. 권장 Dataset Mix

100K 기준 예:

```text
40K  Synthetic defect → Real good
20K  No-op / minimal edit
20K  3D Tooth Reference guided edit
10K  General portrait / face editing
10K  General text-to-image / visual prior
```

Hard case를 oversampling:

- side-ish pose
- low light
- partial smile
- open mouth
- strong specular
- lip occlusion

## 10. Teacher Supervision

두 종류를 섞는다.

### Ground-truth supervision

```text
Input → Real Target
```

### Teacher supervision

```text
Input
 ↓
Teacher prediction
 ↓
Student prediction
```

Teacher만 따라가면 artifact까지 학습할 수 있으므로, 가능하면 실제 GT loss도 유지한다.

## 11. Online vs Offline Distillation

### Online

```text
batch
 ↓
Teacher forward
 ↓
Student forward
 ↓
loss
```

장점:
- arbitrary timestep supervision
- 구현 직관적

단점:
- Teacher+Student 동시 VRAM
- 느림

### Offline Cache

권장.

미리 저장:

```text
source latent
reference latent
text embedding
teacher target/prediction
metadata
```

예:

```text
sample_000001/
├── source_latent.pt
├── reference_latent.pt
├── text_embedding.pt
├── teacher_pred.pt
└── metadata.json
```

Student training 시 Teacher를 GPU에 올리지 않아도 된다.

## 12. Cache 권장 항목

우선:

- VAE image latent
- text embedding
- reference image latent

추가 가능:

- teacher flow/velocity prediction
- teacher intermediate feature
- teacher final latent
- teacher output image

전부 저장하면 저장공간이 커지므로 단계적으로 추가한다.

## 13. Distillation Loss

### 13.1 Flow / Velocity Prediction Loss

가장 먼저 구현.

```python
L_teacher = mse(student_pred, teacher_pred)
```

또는 Huber loss.

### 13.2 Ground Truth Objective

가능하면 원래 flow-matching objective도 유지한다.

```text
L_total =
λ_gt * L_gt
+
λ_teacher * L_teacher
```

### 13.3 Feature Distillation

1B → 0.5B에서 capacity gap이 클 때 고려.

```text
Teacher feature
      ↓ projection
Student feature
```

처음부터 넣지 말고 prediction distillation이 부족할 때 추가한다.

### 13.4 Attention Distillation

Patient ↔ Tooth Reference correspondence를 작은 Student가 놓치는 경우 연구 가치가 있다.

하지만 구현 난이도가 높아 2차 항목으로 둔다.

### 13.5 Identity Preservation

Dental Student에서는 매우 중요.

예:

```text
outside-mouth region
output vs source
```

에 LPIPS / pixel / latent preservation loss를 추가할 수 있다.

## 14. Step Distillation

1B Student가 20~50-step에서 충분히 잘 동작한 뒤 수행한다.

```text
30~50 steps
    ↓
8 steps
    ↓
4 steps
```

Hugging Face Diffusers의 LCM/consistency distillation example은 Stable Diffusion 계열용이지만, 다음 개념을 공부하기 좋다.

- teacher trajectory
- student trajectory
- timestep skipping
- consistency loss
- EMA
- Huber/L2
- boundary conditions

FLUX.2에 그대로 복붙하는 코드는 아니며, custom flow-matching distillation loop가 필요하다.

## 15. 필수 라이브러리

```bash
pip install torch torchvision
pip install diffusers
pip install transformers
pip install accelerate
pip install datasets
pip install safetensors
```

역할:

```text
PyTorch       training / autograd
Diffusers     VAE / scheduler / training reference
Transformers  Qwen/text encoder
Accelerate    mixed precision / multi-GPU
Datasets      dataset pipeline
Safetensors   model weight 저장
```

## 16. 추가 추천 라이브러리

### WebDataset

```bash
pip install webdataset
```

100K~M sample 또는 Teacher cache shard 관리에 유용.

### PEFT

```bash
pip install peft
```

Distillation 자체에는 필수 아님.

유용한 곳:
- Dental Teacher LoRA
- Student LoRA warmup
- adapter experiment

### bitsandbytes

```bash
pip install bitsandbytes
```

optimizer memory 최적화가 필요할 때.

### xFormers / Flash Attention

환경이 맞으면 attention 속도/VRAM 최적화.

### DeepSpeed

멀티GPU, optimizer state가 커질 때 선택.

### LPIPS

```bash
pip install lpips
```

outside-mouth preservation 또는 validation용.

## 17. 별도 Distillation 전용 라이브러리가 필요한가?

필수는 아니다.

핵심은:

```python
with torch.no_grad():
    teacher_pred = teacher(...)

student_pred = student(...)

loss = distillation_loss(
    student_pred,
    teacher_pred
)

accelerator.backward(loss)
optimizer.step()
```

이므로 기본적으로:

```text
PyTorch
+ Diffusers model components
+ Accelerate
```

로 구현 가능하다.

Diffusers consistency distillation example은 training loop 참고용으로 사용한다.

## 18. 참고 Repository

### BFL FLUX.2 Official

```text
https://github.com/black-forest-labs/flux2
```

확인할 부분:

- `Klein4BParams`
- model architecture
- VAE
- Qwen3-4B embedder
- Base vs distilled 설정
- timestep / guidance

### Hugging Face Diffusers

```text
https://github.com/huggingface/diffusers
```

특히:

```text
examples/consistency_distillation/
```

의 LCM training loop를 참고한다.

## 19. 코드 구조 권장

```text
flux2_distill/
├── configs/
│   ├── student_1b.yaml
│   └── student_05b.yaml
├── models/
│   ├── teacher.py
│   └── student.py
├── data/
│   ├── dataset.py
│   └── cache_teacher.py
├── losses/
│   ├── flow_distill.py
│   ├── feature_distill.py
│   └── preserve.py
├── train_distill.py
├── validate.py
└── inference.py
```

## 20. 실험 단계

### Phase 0 — Smoke Test

```text
Student ≈ 1B
Dataset 1K 이하
100~500 steps
```

확인:
- forward
- shape
- finite loss
- backward
- checkpoint

### Phase 1 — 1B Capacity Distillation

```text
Teacher: FLUX.2 Klein 4B Base
Student: ~1B
Dataset: 20K~50K
Inference: 30~50 step 유지
```

먼저 quality를 확보한다.

### Phase 2 — Dental-specific 1B

```text
Teacher: 4B Base + Dental LoRA
Dataset: 50K~200K
```

추가:
- reference edit
- identity preservation
- outside-mouth loss

### Phase 3 — 0.5B

1B recipe를 기반으로:

```text
4B → 0.5B
```

또는:

```text
4B → 1B → 0.5B
```

progressive distillation도 비교한다.

### Phase 4 — Few-step

```text
30~50 → 8 → 4 step
```

각 단계에서 품질/identity/reference adherence/latency/VRAM 측정.

### Phase 5 — Quantization

```text
BF16/FP16
 ↓
INT8
 ↓
INT4
```

최종 배포 모델 크기와 VRAM을 낮춘다.

## 21. 평가 지표

### Image Quality
- LPIPS
- CLIP/DINO similarity
- artifact rate
- human A/B

### Dental
- tooth segmentation consistency
- tooth landmark error
- tooth count
- reference morphology similarity
- shade consistency

### Identity
- face embedding similarity
- face landmark displacement
- outside-mouth LPIPS

### Performance
- parameter count
- weight size
- peak VRAM
- latency
- steps/sec

## 22. 최종 권장 로드맵

```text
FLUX.2 Klein 4B Base
        ↓
Dental LoRA
        ↓
High-quality Teacher
        ↓
1B Dental Student
        ↓
30~50-step quality gate
        ↓
8-step distillation
        ↓
4-step if quality allows
        ↓
0.5B optional experiment
        ↓
INT8 / INT4
```

runtime에서는 text instruction이 제한적이면:

```text
Qwen3-4B
   ↓ offline
cached embeddings
```

으로 바꾸어 text encoder를 제외하는 것도 검토한다.

최종 구조:

```text
Patient image latent
+
optional tooth-reference latent
+
cached task embedding
        ↓
0.5B~1B Dental Student
        ↓
VAE decode
        ↓
Smile Result
```

## 23. 핵심 요약

1. 첫 목표는 0.5B보다 **1B Student**가 현실적이다.
2. **4B→1B capacity distillation**과 **50→4 step distillation**을 분리한다.
3. Dental-specific이면 범용 FLUX capability를 모두 보존할 필요가 없다.
4. 20K~50K로 첫 meaningful experiment, 100K~500K effective samples로 본 실험을 계획한다.
5. Real Good→Synthetic Bad pair와 3D Tooth Reference guided pair를 핵심 데이터로 사용한다.
6. Teacher online inference는 비싸므로 latent/text/teacher prediction cache를 적극 사용한다.
7. 별도 distillation 전용 라이브러리는 필수가 아니며 PyTorch + Diffusers + Accelerate로 구현 가능하다.
8. Diffusers LCM 코드는 FLUX2에 그대로 쓰는 것이 아니라 consistency distillation 구조를 공부하는 reference로 사용한다.
9. 최종 배포에서는 cached text embedding + quantization까지 고려한다.

## 24. FLUX.2 모듈별 Distillation 범위

FLUX.2 Klein 파이프라인은 크게 다음 세 학습/추론 모듈로 구분한다.

```text
Prompt
  ↓
Qwen3-4B Text Encoder       ← Freeze / offline cache
  ↓ text embedding
FLUX.2 Transformer (~4B)    ← Distillation 대상
  ↓ latent prediction
VAE                         ← Freeze / offline cache
  ↓
Image
```

여기서 **Klein 4B의 4B는 핵심 rectified-flow Transformer의 약 4B parameters를 의미한다.** Text Encoder와 VAE까지 합쳐 4B라는 의미가 아니다.

1차 실험에서는 다음처럼 역할을 고정한다.

| Component | Teacher side | Student side | 학습 여부 |
|---|---|---|---|
| Qwen3 Text Encoder | 동일 encoder | 동일 cached embedding 사용 | Freeze |
| VAE Encoder/Decoder | 동일 VAE | 동일 VAE | Freeze |
| FLUX Transformer | 4B | 1B (이후 0.5B) | **Student만 학습** |
| Scheduler / timestep | 동일 조건 | 동일 조건 | parameter 없음 |

즉 첫 실험은 사실상 **FLUX Transformer 4B → Transformer 1B knowledge distillation**이다.

VAE까지 동시에 압축하면 reconstruction error와 Transformer capacity 감소가 섞여 원인 분석이 어려워진다. Text Encoder 압축도 별도 문제로 분리한다.

---

## 25. Base Model 학습과 Inference Step의 관계

Base 모델은 학습 sample 하나마다 inference처럼 50번 denoising하는 방식으로 학습되는 것이 아니다.

일반적인 flow-matching 학습은:

```text
clean latent x0 + noise ε
        ↓
random timestep t
        ↓
x_t
        ↓
model(x_t, t, condition)    # 1 forward
        ↓
flow / velocity target loss
```

처럼 임의의 timestep에서 vector field를 학습한다.

반면 inference의 50-step은 학습된 vector field를 여러 번 평가하여 trajectory를 수치적으로 적분하는 횟수다.

```text
Base model training
random t → model 1 forward → loss

Base inference
Noise → t1 → t2 → ... → t50 → Image
```

따라서 `Base = 50-step으로 학습된 모델`이라기보다 **전체 timestep 영역의 flow field를 학습한 모델이며, 기본 inference에서 약 50번의 model evaluation을 사용하는 모델**로 이해하는 것이 정확하다.

공개된 FLUX.2 Klein 구성에서는 Base가 non-step-distilled 모델이고, production Klein은 timestep/guidance distilled 4-step 모델로 제공된다. 구체적인 BFL 내부 distillation recipe 전체가 공개되어 있다고 가정하지 않는다.

---

## 26. Capacity Distillation과 Step Distillation 분리

두 가지 압축은 별개의 문제다.

```text
                 Capacity Distillation
4B Base  ─────────────────────────→  1B Base-like
~50-step inference                    ~20~50-step inference
                                           │
                                           │ Step Distillation
                                           ↓
                                      1B Fast Student
                                       8 → 4 step
```

### Capacity Distillation

같은 noisy latent와 timestep을 Teacher/Student에 입력한다.

```text
              same x_t, t, text embedding
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Teacher 4B           Student 1B
             │                   │
          pred_T              pred_S
             └─────────┬─────────┘
                       ↓
              KD / flow loss
```

이 단계에서는 inference step 압축을 하지 않는다.

### Step Distillation

1B Base-like 모델의 품질이 확보된 이후:

```text
30~50 step
   ↓
8 step
   ↓
4 step
```

으로 별도 distillation한다.

이 순서를 사용하면 `모델이 작아서 생긴 품질 저하`와 `step을 줄여서 생긴 품질 저하`를 분리해서 분석할 수 있다.

---

## 27. 초기 Dataset 구축 전략: Teacher-generated Data 우선

첫 feasibility test에서는 무작정 웹 크롤링부터 시작하기보다 **현재 확보한 FLUX.2 4B Teacher로 image+prompt pair를 생성하는 방법**이 효율적이다.

```text
Prompt Pool
    ↓
FLUX.2 4B Teacher
    ↓
Generated Image
    ↓
(image, prompt) dataset
```

장점:

- Teacher가 잘 표현하는 distribution을 Student가 직접 모사할 수 있음
- caption mismatch가 적음
- watermark / 저화질 데이터 정제가 줄어듦
- 첫 distillation feasibility 판단이 쉬움
- 데이터 사용권과 provenance 관리가 상대적으로 단순해짐(단, 사용한 모델/입력/출력의 라이선스와 내부 정책은 별도 확인)

하지만 최종 Dental Student는 teacher-generated image만으로 구성하지 않는다.

권장 progression:

```text
Phase 0
Teacher-generated 100%
5K~10K
→ pipeline / feasibility

Phase 1
Teacher-generated 중심
+ real dental/smile
+ synthetic dental pairs
20K~50K

Phase 2
Dental-specific data 중심
50K~200K+
```

최종 목적이 patient image editing이므로 실제 domain distribution과 Real Good → Synthetic Bad pair를 반드시 추가한다.

---

## 28. Distillation Dataset에 최소 필요한 정보

첫 capacity distillation은 dataset 관점에서 기본적으로:

```text
image + prompt
```

만 있어도 시작할 수 있다.

학습 preprocessing에서:

```text
image
 ↓ frozen VAE encoder
x0 latent

prompt
 ↓ frozen Qwen Text Encoder
text embedding
```

을 만든다.

학습 iteration에서는:

```text
cached x0
+ random noise
+ random timestep t
        ↓
x_t
        ↓
Teacher 4B / Student 1B
        ↓
prediction matching
```

따라서 단순 text-to-image capacity distillation의 최소 원본 dataset format은 다음 정도면 충분하다.

```text
image.png
prompt.txt
```

Dental image editing 단계에서는 추가로:

```text
source image
instruction/prompt
target image
optional tooth reference image
optional mouth/teeth mask
```

를 사용한다.

---

## 29. Offline Cache 전략

Distillation에서는 **Text Encoder와 VAE를 매 iteration마다 다시 실행하지 않는 것을 권장**한다.

특히 Qwen3-4B Text Encoder는 Student보다도 클 수 있으므로 매 batch forward는 낭비가 크다.

### 권장 preprocessing

```text
                 OFFLINE PREPROCESS

prompt ──→ Qwen3-4B ──→ text embedding ──→ cache

image  ──→ VAE encoder ─→ image latent ───→ cache

reference image
       └─→ VAE encoder ─→ reference latent → cache
```

그 후 training은:

```text
cached image latent
+
cached text embedding
+
random noise/timestep
        ↓
Teacher 4B forward
        ↓
Student 1B forward/backward
```

만 수행한다.

### 권장 Cache 구조

```text
sample_000001/
├── image_latent.pt
├── text_embedding.pt
├── pooled_embedding.pt       # model interface에서 필요할 경우
├── reference_latent.pt       # reference edit일 경우
├── prompt.txt                # debugging/provenance
└── metadata.json
```

원본 image/prompt도 가능하면 별도로 보존한다. embedding format이나 VAE가 변경될 경우 cache를 다시 생성할 수 있어야 하기 때문이다.

### Cache dtype

초기 후보:

```text
BF16 또는 FP16
```

A/B 검증 후 precision을 고정한다. 저장 공간이 충분하고 numerical baseline이 필요하면 소규모 FP32 cache와 비교한다.

---

## 30. Teacher Prediction은 처음에는 Cache하지 않는다

Teacher prediction은 다음에 의존한다.

```text
image latent
text embedding
noise ε
timestep t
```

따라서 같은 image/prompt라도:

```text
t=0.1
t=0.5
t=0.9
```

에서 prediction이 모두 다르다.

Teacher prediction을 완전히 offline cache하려면:

```text
sample
× multiple timesteps
× multiple noises
```

를 저장해야 하므로 storage가 급격히 증가하고 timestep/noise sampling 자유도도 떨어진다.

따라서 첫 구현은:

```text
Image latent       CACHE
Text embedding     CACHE
Reference latent   CACHE (필요 시)
Teacher prediction ONLINE
```

을 권장한다.

Teacher forward는 `torch.no_grad()`로 실행한다.

```python
with torch.no_grad():
    teacher_pred = teacher(x_t, t, text_embedding)

student_pred = student(x_t, t, text_embedding)
loss = distill_loss(student_pred, teacher_pred)
```

---

## 31. Teacher-generated Image와 Knowledge Distillation의 차이

두 개념을 구분한다.

### Synthetic Dataset Generation

```text
prompt
 ↓
Teacher 4B full inference
 ↓
image
 ↓
(image, prompt) 저장
```

이 단계는 **학습 데이터 distribution을 구축하는 과정**이다.

### Actual Knowledge Distillation

```text
saved image → x0
prompt      → embedding

random t/noise → x_t

x_t ──→ Teacher 4B ──→ pred_T
 │
 └───→ Student 1B ──→ pred_S

loss(pred_S, pred_T)
```

즉 `Teacher가 만든 최종 이미지로 supervised training`하는 것만으로 끝내지 않고, **동일한 noisy latent/timestep에서 Teacher의 flow prediction을 Student가 따라가도록 하는 것**을 핵심 KD signal로 사용한다.

가능하면 실제 ground-truth objective도 함께 사용한다.

```text
L_total = λ_gt * L_gt
        + λ_kd * L_teacher
```

---

## 32. 업데이트된 1차 실험 Recipe

가장 먼저 구현할 실험을 최대한 단순하게 고정한다.

### Dataset

```text
5K~10K image + prompt
```

초기에는 FLUX.2 4B로 생성한 synthetic dataset을 사용해도 된다.

### Offline preprocessing

```text
Qwen3-4B → text embedding cache
VAE       → image latent cache
```

### Models

```text
Qwen3-4B         Frozen / training runtime에서 제거 가능
VAE              Frozen
Teacher FLUX 4B  Frozen + no_grad
Student FLUX 1B  Train
```

### Per iteration

```text
1. cached latent x0 load
2. cached text embedding load
3. random noise ε 생성
4. random timestep t sampling
5. x_t 생성
6. Teacher 4B forward (no_grad)
7. Student 1B forward
8. flow/KD loss
9. Student backward/update
```

### 초기에는 하지 않는 것

```text
VAE training                X
Text Encoder training       X
Teacher prediction cache    X
0.5B부터 시작               X
4-step distillation 동시진행 X
INT4 quantization           X
```

### 첫 성공 조건

```text
Student ≈ 1B
20~50 inference steps
Teacher와 동일 prompt/seed 조건 비교
```

에서 생성 품질과 prompt adherence가 충분히 유지되는지 확인한다.

성공 후 Dental-specific data를 추가하고, 그 다음 0.5B 및 few-step distillation로 진행한다.

---

## 33. 현재 권장 전체 순서

```text
[0] FLUX.2 4B Base/Teacher 준비
            ↓
[1] Prompt pool 구축
            ↓
[2] 4B로 synthetic image 생성
            ↓
[3] image + prompt dataset 확보
            ↓
[4] Qwen embedding offline cache
    VAE latent offline cache
            ↓
[5] 4B Teacher → 1B Student
    capacity distillation
    Teacher prediction은 online
            ↓
[6] 20~50-step quality gate
            ↓
[7] Real/Synthetic Dental data 추가
            ↓
[8] Dental-specific 1B Student
            ↓
[9] 0.5B capacity experiment
            ↓
[10] 8-step distillation
            ↓
[11] 4-step distillation
            ↓
[12] INT8 / INT4 deployment optimization
```

이 순서의 핵심은 **한 실험에서 하나의 압축 축만 변경하여 실패 원인을 분리하는 것**이다.

---

## References

- Black Forest Labs FLUX.2 official repository  
  https://github.com/black-forest-labs/flux2

- Hugging Face Diffusers  
  https://github.com/huggingface/diffusers

- Diffusers consistency distillation examples  
  https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation
