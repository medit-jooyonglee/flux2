# LoRA Training for FLUX.2

This repository is an **inference-only** codebase. There is no training loop, optimizer, loss computation, dataset abstraction, or LoRA/PEFT integration anywhere in `src/flux2` or `scripts/` — only forward-pass sampling and checkpoint loading. This doc summarizes what exists today, why LoRA can't be trained directly against this codebase, and which external, verified training stacks to use instead.

## Current state of this repo

- [`src/flux2/model.py`](../src/flux2/model.py) defines the `Flux2` DiT (`DoubleStreamBlock`, `SingleStreamBlock`, `SelfAttention`, etc.) — inference-only, no gradient/training code.
- [`src/flux2/sampling.py`](../src/flux2/sampling.py) implements the flow-matching denoising loops (`denoise`, `denoise_cached`, `denoise_cfg`) — forward passes only, no backward/loss.
- [`src/flux2/util.py`](../src/flux2/util.py) builds the model on `torch.device("meta")` and loads a full checkpoint via `load_state_dict(sd, strict=True, assign=True)` — a strict, monolithic loader with no notion of adapter weights.
- `pyproject.toml` depends on `torch`, `transformers`, `accelerate`, `einops`, `safetensors` — **no `diffusers`, `peft`, `xformers`, or any optimizer/scheduler libs**.
- Repo-wide search for `lora|LoRA|requires_grad|optimizer|backward()|DataLoader|train_step` turns up nothing in code; `README.md` only mentions LoRA training as a reason to prefer the Base checkpoints.

### Why this architecture is awkward for off-the-shelf LoRA tooling

Unlike diffusers' `Attention` module (separate `to_q`/`to_k`/`to_v`/`to_out` linears), this repo fuses projections into single matrices:

- `SelfAttention.qkv`: `nn.Linear(dim, dim*3, bias=False)`
- `SingleStreamBlock.linear1`: QKV **and** MLP-in fused into one matrix (`hidden*3 + mlp_hidden*2`)
- `SingleStreamBlock.linear2`: attention-out **and** MLP-out fused into one matrix

PEFT's automatic target-module name matching (which expects `to_q`/`to_k`/etc.) does not work against this naming scheme without a custom LoRA wrapper that knows how to split/inject into these fused matrices.

## Recommended path: train elsewhere, convert weights

Rather than building a training loop against this repo's custom model definition, train against one of the verified external stacks below (which target **diffusers' own** `FluxTransformer2DModel`-style implementation of FLUX.2), then map the resulting LoRA weight names onto this repo's fused layer names before loading for inference here.

### 1. Diffusers official DreamBooth LoRA scripts

- [`train_dreambooth_lora_flux2.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2.py) — FLUX.2 [dev]/[klein], text-to-image
- [`train_dreambooth_lora_flux2_klein.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py) — klein-specific variant
- [`train_dreambooth_lora_flux2_img2img.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2_img2img.py) — image editing, trains on (condition image, target image, instruction text) triplets
- Docs: [`README_flux2.md`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md)

Supported variants: FLUX.2 [dev] (Mistral Small 3.1 text encoder, memory-intensive) and FLUX.2 [klein] 4B/9B (Qwen VL text encoder, much lighter).

Key CLI args: `--rank`, `--lora_alpha`, `--lora_layers` (target modules), `--learning_rate` (1e-4 default, 1.0 with Prodigy), `--resolution`, `--cache_latents`, FP8/NF4 quantization (FP8 needs CUDA compute capability 8.9+), gradient checkpointing, CPU offloading.

Note: klein does not support `--remote_text_encoder` — the Qwen VL text encoder must be loaded locally (offloading is still supported).

### 2. AI-Toolkit

Currently described as the de-facto reference implementation for FLUX.2 training. Provides a web UI and runs FLUX.2 [klein] 4B in ~12GB VRAM and 9B in ~16GB+ (scaled-down `train_lora_flux_24gb.yaml`).

### 3. Kohya / musubi-tuner

`sd-scripts` (Kohya) officially supports only FLUX.1. FLUX.2/klein LoRA training reportedly already works in musubi-tuner (per a user report in [issue #890](https://github.com/kohya-ss/musubi-tuner/issues/890), opened 2026-02-04), but full fine-tuning support there is unconfirmed/unmaintained as of that issue.

## Training data format

- Standard LoRA: (image, caption) pairs. Style/concept LoRAs can work with as few as 20–a few hundred images.
- Image editing / conditioning LoRA: (condition image, target image, instruction text) triplets — see `kontext-community/relighting` on the Hub for a reference dataset shape.
- Resolution should match the base model's expected resolution; latents can be precomputed and cached (`--cache_latents`) to speed up training.

## Using a trained LoRA with this repo

The LoRA weights produced by the scripts above use diffusers' naming convention (`to_q`, `to_k`, `to_v`, `to_out`, feed-forward linears, etc.), which does **not** match this repo's fused `qkv`/`proj`/`linear1`/`linear2` names. To use such a LoRA for inference here, either:

1. Write a conversion script that maps/splits the diffusers LoRA A/B matrices onto this repo's fused linear layers and merges or injects them at load time (a custom loader alongside [`util.py`](../src/flux2/util.py), since the existing loader is `strict=True` and rejects extra adapter keys), or
2. Run inference through the diffusers pipeline directly instead of this repo's `Flux2`/`sampling.py` path.

## Hardware notes

`Flux2.dev` is a 32B-parameter model; even LoRA training needs FSDP/DeepSpeed-class infrastructure or heavy quantization — `accelerate` alone (already a dependency here, but unused for training) is not sufficient at that scale. **FLUX.2 [klein] 4B/9B Base are the realistic LoRA targets** given typical consumer/workstation VRAM budgets.
