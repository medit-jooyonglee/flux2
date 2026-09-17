# FLUX.2 Klein 4B → ~1B capacity distillation

Implements Phase 0 (smoke test) + Phase 1 (capacity distillation) of
[`../docs/flux2_distillation_05b_1b.md`](../docs/flux2_distillation_05b_1b.md) (see its
section 32, "업데이트된 1차 실험 Recipe", for the exact recipe this follows).

## Does this need a separate training script?

Yes. There is no official script for this: diffusers' own FLUX.2 examples only cover LoRA
fine-tuning (adapters on top of the *existing* 4B/9B architecture) — nothing compresses
the architecture itself to a smaller size. That's a genuinely different problem (full
parameter training of a custom smaller student model), so it needs a custom loop.

The good news: unlike the LoRA case (`../docs/flux2_lora_training.md`), this doesn't hit
the "fused qkv/linear1/linear2 layer naming blocks PEFT" problem at all -- there's no
adapter injection here, we're training *all* of the (smaller) Student's own parameters
directly, so there's nothing for a naming convention to conflict with. That's why this
uses this repo's own `Flux2` class directly instead of diffusers' reimplementation.

## What's official/library-based vs. custom here

| Piece | Implementation |
|---|---|
| Training loop, autograd, checkpointing | PyTorch (official) |
| Mixed precision, device placement | `accelerate` (official) |
| Frozen VAE / Qwen3-4B text encoder / Flux2 transformer | this repo's own `src/flux2` (already used for inference) |
| Teacher/Student construction, KD loss | custom (`student.py`, `train_distill.py`) -- this is the part that has to be custom, everywhere else reuses existing library/repo code |

No `diffusers`, no `peft` needed for this part (they matter for the *LoRA* path in
`../training/`, not here).

## Files

- `prepare_cache.py` -- offline preprocessing (doc section 29): encodes each (image,
  prompt) pair through the frozen VAE + Qwen3-4B text encoder once, caches
  `latent.pt`/`embedding.pt`. Training never touches either model afterward.
- `cache_dataset.py` -- reads the cache directory as a plain `torch.utils.data.Dataset`.
- `student.py` -- builds the Teacher (`flux.2-klein-base-4b`, frozen) and the Student.
  Per doc section 5/6, the Student keeps the *same* `hidden_size`/`num_heads`/
  `context_in_dim` as the Teacher and only reduces depth -- `Klein4BParams(depth=2,
  depth_single_blocks=3)`, measured at **1.054B params** (`python student.py` prints this).
  Because every non-block submodule then has an identical shape to the Teacher's, they're
  copied 1:1; `double_blocks`/`single_blocks` are initialized from evenly-spaced Teacher
  blocks (section 6-B's "Teacher layer mapping").
- `train_distill.py` -- the main script. Per step: sample noise + a random timestep `t`,
  build the flow-matching `x_t`, run Teacher (`no_grad`) and Student on the *same* `x_t`/
  `t`/text embedding, and combine the ground-truth flow loss with the teacher-matching
  (KD) loss (section 13.2/31). Teacher prediction is **not** cached (section 30 -- it
  depends on the sampled noise/timestep, which are different every step).

## VRAM (measured on a 48GB A6000)

| batch_size | validation this step? | peak allocated | peak reserved |
|---|---|---|---|
| 1 | no | 29.6 GB | 30.7 GB |
| 2 | no | ~30 GB | ~31 GB |
| 2 | yes (loads VAE + generates images) | 34.5 GB | 42.6 GB |

Baseline (~30GB at batch_size=1) is mostly fixed cost: Teacher 3.9B in bf16 (~7.8GB,
frozen so no optimizer state) + Student 1.05B in fp32 (~4.2GB weights + ~8.4GB AdamW
momentum/variance, since mixed precision keeps fp32 master weights) + activations for
both models' forward passes. Validation adds the (frozen) VAE plus multi-step denoise
activations on top -- budget for it, don't assume the no-validation number is your ceiling.

On this hardware, batch_size=2 with validation on is already using ~85% of a 48GB card's
reserved memory -- there's headroom for batch_size=1-2 per GPU but not much more without
gradient checkpointing or a larger GPU. Two assigned GPUs are better spent running
one process per GPU (via `accelerate launch --multi_gpu` once you want that) for more
throughput/larger effective batch, not for fitting a bigger single-process batch.

## Smoke test (done -- Phase 0)

```bash
python prepare_cache.py --data_dir=/data1/jooyonglee/smiledesign \
  --output_dir=/data1/jooyonglee/smiledesign_cache_smoke --limit=30

CUDA_VISIBLE_DEVICES=<a free GPU> python train_distill.py \
  --cache_dir=/data1/jooyonglee/smiledesign_cache_smoke \
  --output_dir=/data1/jooyonglee/smiledesign_runs/smoke \
  --max_steps=20 --batch_size=2 --log_every=1
```

Result: 30 samples cached, 20 steps ran (student 1.054B params, teacher 3.876B),
finite loss every step, checkpoint saved and reloadable
(`/data1/jooyonglee/smiledesign_runs/smoke/student_final.pt`, verified to contain
1.054B params across 45 tensors). This machine's GPUs are shared with other jobs --
`CUDA_VISIBLE_DEVICES` had to point at whichever GPU actually had free memory
(`nvidia-smi` first) rather than assuming GPU 0.

## Validation images (TensorBoard)

`pip install tensorboard` is required -- `accelerate` silently no-ops `log_with="tensorboard"`
(no error, just never writes anything) if the package isn't there, so double check it's
installed if a run's `<output_dir>/distill/` has no `events.out.tfevents.*` file.

Every `--validation_every_epochs` epochs (default: 1, epoch = one full pass over the
cached dataset given `--batch_size`), `--num_validation_images` (default 4) fixed samples
from the cache are run through the *actual* multi-step denoise loop (`--validation_num_steps`,
default 30, matching the doc's Phase 1 target) with the Student's current weights, decoded
through the VAE, and logged to TensorBoard as `validation/student`. The Teacher's own
output on the same samples/seeds is logged once at step 0 as `validation/teacher_reference`
so you can compare Student progress against it directly -- this is exactly the doc's
"첫 성공 조건: Teacher와 동일 prompt/seed 조건 비교" (section 32). Scalars (`train/loss`,
`train/loss_gt`, `train/loss_kd`) are logged every step too.

```bash
tensorboard --logdir=/data1/jooyonglee/smiledesign_runs/smoke_val
```

Caveat: the validation samples are drawn from the training cache itself (no held-out
split exists yet) -- treat this as a "is it visibly learning / not broken" sanity check,
not a real quality metric. Pass `--validation_every_epochs=0` to disable it entirely.

## Next: Phase 1 (real run)

Once the full ~10K-image upload to `/data1/jooyonglee/smiledesign/` finishes:

```bash
python prepare_cache.py --data_dir=/data1/jooyonglee/smiledesign \
  --output_dir=/data1/jooyonglee/smiledesign_cache   # no --limit: cache everything

python train_distill.py \
  --cache_dir=/data1/jooyonglee/smiledesign_cache \
  --output_dir=/data1/jooyonglee/smiledesign_runs/phase1 \
  --max_steps=<a few thousand+> --batch_size=<as large as VRAM allows> \
  --save_every=500
```

Doc section 20 (Phase 1) targets 20K-50K samples and expects inference-time quality to
be checked at 30-50 steps (this script only trains; comparing Teacher vs. Student output
quality side-by-side still needs a `validate.py`/inference script -- not built yet, since
Phase 0/1 here only covers *training* runs correctly, not a quality gate).

Not implemented yet, deliberately, per the doc's own "초기에는 하지 않는 것" list:
feature/attention distillation, identity-preservation loss, teacher-prediction caching,
step (4/8-step) distillation, 0.5B, quantization.
