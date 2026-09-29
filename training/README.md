# Smile-design LoRA training environment

Concrete, runnable PEFT/LoRA training setup for fine-tuning FLUX.2 [klein] 4B on the
dental smile-design task. This lives entirely in `training/` and its own virtualenv --
it does **not** touch `src/flux2` or the environment `scripts/bench_klein4b.py`/
`scripts/server.py` run in (see "Why a separate venv" below).

## Does freezing text_encoder + VAE and training only the DiT make sense?

Yes -- that's exactly what the official script does, unconditionally:

```
# train_dreambooth_lora_flux2_klein_img2img.py:1182,1185-1186
text_encoder.requires_grad_(False)
transformer.requires_grad_(False)   # LoRA adapters are then added on top of this
vae.requires_grad_(False)
```

There's no flag to enable training the text encoder in this script at all. Rationale:
the text encoder (Qwen3) and VAE are general-purpose (language understanding / pixel
<-> latent mapping) and don't need to change for a narrower visual-style task; only the
denoising transformer needs to learn the new condition_image -> target_image mapping.
This also keeps trainable parameter count tiny (LoRA rank 16 on attention/MLP
projections), which is why 2x48GB is comfortably enough (see below).

## Layout

```
training/
  .venv/                    isolated Python env (diffusers/peft/accelerate/etc.)
  external/                 unmodified upstream diffusers dreambooth LoRA scripts
                            (fetched from huggingface/diffusers@04e724c)
  accelerate_config_2gpu.yaml
  launch_lora_train.sh      training launch wrapper
  dataset/                  data construction (see dataset/README.md)
    build_dataset.py
    smile_design/           the actual HF-`datasets`-loadable training set (gitignored)
  eval/
    evaluate.py             scoring + human-review scaffolding for a trained LoRA
```

## Why a separate venv (don't skip this)

Installing `diffusers` into the system/repo Python broke this repo's own inference stack
the first time around: `pip install diffusers` pulled `huggingface-hub>=1.0`, which is
incompatible with this repo's pinned `transformers==4.56.1` (`transformers` refused to
import at all afterward). `training/.venv` is a **fully independent** virtualenv (not
`--system-site-packages`) with its own torch/transformers/diffusers stack, so this can
never happen again regardless of what training-side dependencies need. Always
`source training/.venv/bin/activate` before running anything in this directory; never
`pip install` diffusers/peft/etc. outside of it.

## Setup (already done once; here for rebuilding)

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/huggingface/diffusers.git@04e724ce518e087301052b65ff81ec35d54f4040" \
  accelerate peft datasets bitsandbytes prodigyopt ftfy tensorboard sentencepiece protobuf transformers \
  insightface onnxruntime-gpu lpips
```
(`external/` was populated via a sparse checkout of `examples/dreambooth` from that same
diffusers commit -- pinned deliberately since these training scripts are new/actively
changing on `main`.)

## Workflow

1. **Build the dataset** -- see [`dataset/README.md`](dataset/README.md) for the full
   guide (bootstrapping candidates from generic faces + a style vocabulary, curating
   them, optionally mixing in real before/after pairs).

2. **Train**:
   ```bash
   CUDA_VISIBLE_DEVICES=<your 2 assigned GPU indices> \
     ./launch_lora_train.sh ./dataset/smile_design
   ```
   Defaults: rank 16, bf16, `--cache_latents` + `--gradient_checkpointing` +
   `--use_8bit_adam` to fit comfortably in 48GB/GPU, 2000 steps, checkpoint every 250.
   No `--do_fp8_training`: these are Ampere (RTX A6000, cc 8.6) GPUs, and FP8 training/
   inference needs cc >= 8.9 -- we already hit this exact wall getting FP8 *inference*
   working earlier in this repo (`src/flux2/text_encoder.py`'s `_fp8_supported()`).
   Monitor with `tensorboard --logdir training/output/smile-design-lora/logs`.

3. **Evaluate** -- see [`eval/evaluate.py`](eval/evaluate.py)'s module docstring. Runs
   the trained LoRA over a held-out manifest and reports: face-identity preservation
   (ArcFace cosine similarity), instruction/style alignment (CLIPScore-style), and
   optionally LPIPS similarity to a real target when you have one. Also emits a
   `manual_review.csv` template -- **automatic metrics here are proxies, not a
   pass/fail gate**; a human (ideally dental-trained) rating of realism and clinical
   plausibility is still required before calling a checkpoint good.

## Using the trained LoRA elsewhere

The LoRA weights use diffusers' naming (`to_q`/`to_k`/`to_v`/`to_out`/etc.), which does
**not** match this repo's fused `qkv`/`linear1`/`linear2` layer names in
`src/flux2/model.py`. To actually serve results (e.g. from `scripts/server.py`), either:
- keep serving through `Flux2KleinPipeline` (diffusers) directly, same as `eval/evaluate.py`, or
- write a conversion script mapping the diffusers LoRA A/B matrices onto this repo's
  fused layers (not built yet -- worth doing once a LoRA is actually good enough to ship).

See [`../docs/flux2_lora_training.md`](../docs/flux2_lora_training.md) for the earlier
survey this setup is based on.
