"""Minimal inference / latency benchmark for FLUX.2 [klein] 4B.

Unlike scripts/cli.py, this does NOT load the FLUX.2-dev (Mistral-Small-24B) text
encoder used there for prompt/image moderation and prompt upsampling -- that model
is unrelated to klein 4B's own generation and needs ~48GB just for its weights,
which won't fit on consumer GPUs. This script only loads what klein 4B actually
needs to run: the flow model, the autoencoder, and the Qwen3-4B text encoder.

Models are loaded once via a module-level singleton (get_models) so repeated calls
to infer() -- from main(), main_test(), or your own code importing this module --
don't pay the disk-load cost again. By default (--cpu_offloading=True) infer() also
shuttles the text encoder / flow model between CPU and GPU around each call so the
three models never have to share GPU memory at once -- on a 16GB card, keeping all
three resident (~13GB of weights alone, plus denoise/decode activations) pushes peak
usage past physical VRAM and silently triggers Windows' CUDA system-memory fallback,
which is dramatically slower than an OOM. Pass --cpu_offloading=False to keep
everything on GPU throughout (faster per call, needs more VRAM headroom).

Usage:
  # single prompt, repeated timing runs (same as before)
  PYTHONPATH=src python scripts/bench_klein4b.py main --prompt="a cat in a hat" --runs=5

  # several different prompts in one process, models loaded only once
  PYTHONPATH=src python scripts/bench_klein4b.py main_test --prompts="a cat|a dog|a bird"
"""

import time
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    cap_size,
    denoise,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

MODEL_NAME = "flux.2-klein-4b"
MODEL_INFO = FLUX2_MODEL_INFO[MODEL_NAME]
DEVICE = torch.device("cuda")
MAX_GEN_PIXELS = 1024 * 1024  # klein 4B's native training resolution budget

_MODELS: dict[str, object] | None = None


def get_models(cpu_offloading: bool = True) -> dict[str, object]:
    """Singleton loader for the text encoder / flow model / autoencoder.

    Loads once per process; later calls (even with a different cpu_offloading value)
    just return the already-loaded models.
    """
    global _MODELS
    if _MODELS is None:
        print("Loading text encoder (Qwen3-4B-FP8)...")
        text_encoder = load_text_encoder(MODEL_NAME, device=DEVICE).eval()
        print("Loading flow model (klein 4B)...")
        model = load_flow_model(MODEL_NAME, device="cpu" if cpu_offloading else DEVICE).eval()
        print("Loading autoencoder...")
        ae = load_ae(MODEL_NAME, device=DEVICE).eval()
        _MODELS = {"text_encoder": text_encoder, "model": model, "ae": ae}
    return _MODELS


def _load_images(input_images: str) -> list[Image.Image]:
    if not input_images:
        return []
    paths = [p.strip() for p in input_images.split(",") if p.strip()]
    return [Image.open(p).convert("RGB") for p in paths]


def _resolve_size(
    width: int, height: int, img_ctx: list[Image.Image], match_image_size: int | None
) -> tuple[int, int]:
    if match_image_size is not None:
        if match_image_size < 0 or match_image_size >= len(img_ctx):
            print(
                f"  ! match_image_size={match_image_size} is out of range "
                f"(0-{len(img_ctx) - 1}). Using given width/height: {width}x{height}"
            )
        else:
            width, height = img_ctx[match_image_size].size
            print(f"Matched dimensions from image {match_image_size}: {width}x{height}")

    orig_width, orig_height = width, height
    width, height = (width // 16) * 16, (height // 16) * 16
    assert width > 0 and height > 0, "width/height must be at least 16"
    if (width, height) != (orig_width, orig_height):
        print(f"  ! Rounded dimensions down to a multiple of 16: {orig_width}x{orig_height} -> {width}x{height}")
    return width, height


def _to_pil(x: torch.Tensor) -> Image.Image:
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


def infer(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    match_image_size: int | None = None,
    num_steps: int | None = None,
    guidance: float | None = None,
    seed: int = 0,
    input_images: str = "",
    cpu_offloading: bool = True,
) -> tuple[Image.Image, float]:
    """Run one klein 4B generation and return (image, elapsed_seconds).

    Does not save anything -- callers (main(), main_test(), or your own code) decide
    where and how to persist the result.
    """
    models = get_models(cpu_offloading=cpu_offloading)
    text_encoder, model, ae = models["text_encoder"], models["model"], models["ae"]

    num_steps = num_steps if num_steps is not None else MODEL_INFO["defaults"]["num_steps"]
    guidance = guidance if guidance is not None else MODEL_INFO["defaults"]["guidance"]

    img_ctx = _load_images(input_images)
    out_width, out_height = _resolve_size(width, height, img_ctx, match_image_size)

    # Generate at a resolution capped to MAX_GEN_PIXELS (keeping aspect ratio) -- klein 4B
    # is trained around 1024x1024, and denoise/decode cost scales with pixel count, so an
    # oversized request (e.g. width/height taken from a UHD match_image_size reference)
    # would otherwise be slow or OOM. The output is upscaled back to out_width/out_height
    # (the originally requested/matched size) after decoding.
    width, height = cap_size(out_width, out_height, MAX_GEN_PIXELS)
    width, height = (width // 16) * 16, (height // 16) * 16
    if (width, height) != (out_width, out_height):
        print(f"  ! Capping generation to {width}x{height} (from {out_width}x{out_height}); will upscale back after")

    with torch.no_grad():
        # text_encoder is always on GPU on entry (every exit path below restores it there,
        # or never moved it in the first place). model's device, however, depends on
        # whichever cpu_offloading value the *previous* infer() call (or get_models(), for
        # the first call) used -- since get_models() is a singleton, a call with
        # cpu_offloading=False can't assume model is already on GPU just because this call
        # didn't ask for offloading. So model.to(DEVICE) below must be unconditional.
        ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)
        ctx = text_encoder([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        if cpu_offloading:
            text_encoder.cpu()
            torch.cuda.empty_cache()
        model.to(DEVICE)

        gen_device = next(model.parameters()).device
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=gen_device)
        x, x_ids = batched_prc_img(randn)
        timesteps = get_schedule(num_steps, x.shape[1])

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = denoise(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        out = ae.decode(x).float()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if cpu_offloading:
            model.cpu()
            torch.cuda.empty_cache()
            text_encoder.to(DEVICE)

    result_img = _to_pil(out)
    if (width, height) != (out_width, out_height):
        result_img = result_img.resize((out_width, out_height), Image.Resampling.LANCZOS)
    return result_img, elapsed


def main(
    # prompt: str = "a photo of a forest with mist swirling around the tree trunks",
    prompt: str = "a pretty girl",
    width: int = 1024,
    height: int = 1024,
    match_image_size: int | None = None,
    num_steps: int | None = None,
    guidance: float | None = None,
    seed: int = 0,
    input_images: str = "",
    warmup: int = 1,
    runs: int = 3,
    output_dir: str = "output",
    cpu_offloading: bool = True,
):
    """Benchmark: repeat the same prompt `runs` times (different seed each time), report latency stats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Warmup ({warmup} run(s))...")
    for _ in range(warmup):
        infer(
            prompt,
            width,
            height,
            match_image_size,
            num_steps,
            guidance,
            seed,
            input_images,
            cpu_offloading,
        )

    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(runs):
        img, elapsed = infer(
            prompt,
            width,
            height,
            match_image_size,
            num_steps,
            guidance,
            seed + i,
            input_images,
            cpu_offloading,
        )
        times.append(elapsed)
        img.save(out_dir / f"bench_klein4b_{i}.png")
        print(f"  run {i + 1}/{runs}: {elapsed * 1000:.1f} ms")

    avg = sum(times) / len(times)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(
        f"\navg {avg * 1000:.1f} ms  (min {min(times) * 1000:.1f} ms, max {max(times) * 1000:.1f} ms)\n"
        f"peak GPU memory: {peak_mem:.2f} GB\n"
        f"saved images to {out_dir}/"
    )


def main_test(
    prompts: str='a pretty  girl',
    width: int = 1024,
    height: int = 1024,
    match_image_size: int | None = None,
    num_steps: int | None = None,
    guidance: float | None = None,
    seed: int = 0,
    input_images: str = "",
    output_dir: str = "output",
    cpu_offloading: bool = True,
):
    """Run several different prompts in one process (models loaded only once via the singleton).

    prompts: '|'-separated list, e.g. --prompts="a cat|a dog wearing sunglasses|a red bicycle"
    Each prompt gets seed+i. Saving happens here -- edit this loop to change filenames/format
    or to do something other than save (e.g. collect images in memory) as you see fit.
    """
    prompt_list = [p.strip() for p in prompts.split("|") if p.strip()]
    assert prompt_list, "no prompts given (use --prompts=\"a|b|c\")"

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    for i, prompt in enumerate(prompt_list):
        img, elapsed = infer(
            prompt=prompt,
            width=width,
            height=height,
            match_image_size=match_image_size,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed + i,
            input_images=input_images,
            cpu_offloading=cpu_offloading,
        )
        path = out_dir / f"test_{i}.png"
        img.save(path)  # <-- customize this: path, format, or skip saving entirely
        print(f"[{i + 1}/{len(prompt_list)}] {elapsed * 1000:.1f} ms  prompt={prompt!r} -> {path}")


if __name__ == "__main__":
    import torch
    torch.cuda.set_device('cuda:5')
    # from fire import Fire

    # Fire({"main": main, "main_test": main_test})
    main_test(
        cpu_offloading=False
    )
