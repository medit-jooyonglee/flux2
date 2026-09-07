"""Minimal inference / latency benchmark for FLUX.2 [klein] 4B.

Unlike scripts/cli.py, this does NOT load the FLUX.2-dev (Mistral-Small-24B) text
encoder used there for prompt/image moderation and prompt upsampling -- that model
is unrelated to klein 4B's own generation and needs ~48GB just for its weights,
which won't fit on consumer GPUs. This script only loads what klein 4B actually
needs to run: the flow model, the autoencoder, and the Qwen3-4B text encoder.

By default this also offloads the text encoder to CPU (--cpu_offloading=True) right
after it encodes the prompt, so the flow model never has to share GPU memory with it
during denoise+decode. On a 16GB card, keeping all three models resident at once
(~13GB of weights alone) pushes peak usage past physical VRAM and silently triggers
Windows' CUDA system-memory fallback, which is dramatically slower than an OOM. Pass
--cpu_offloading=False to disable this and keep everything on GPU throughout.

Usage:
  PYTHONPATH=src python scripts/bench_klein4b.py
  PYTHONPATH=src python scripts/bench_klein4b.py --prompt="a cat in a hat" --width=1024 --height=1024 --runs=5
  PYTHONPATH=src python scripts/bench_klein4b.py --input_images="ref1.png,ref2.png"
"""

import time
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, encode_image_refs, get_schedule, scatter_ids
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

MODEL_NAME = "flux.2-klein-4b"


def _load_images(input_images: str) -> list[Image.Image]:
    if not input_images:
        return []
    paths = [p.strip() for p in input_images.split(",") if p.strip()]
    return [Image.open(p).convert("RGB") for p in paths]


def _save(x: torch.Tensor, path: Path):
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(path)


def main(
    # prompt: str = "a photo of a forest with mist swirling around the tree trunks",
    prompt: str = "a pretty woman scarllet yohanson",
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
    model_info = FLUX2_MODEL_INFO[MODEL_NAME]
    device = torch.device("cuda")

    num_steps = num_steps if num_steps is not None else model_info["defaults"]["num_steps"]
    guidance = guidance if guidance is not None else model_info["defaults"]["guidance"]

    img_ctx = _load_images(input_images)

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

    print(f"Loading text encoder (Qwen3-4B-FP8)...")
    text_encoder = load_text_encoder(MODEL_NAME, device=device).eval()
    print("Loading flow model (klein 4B)...")
    model = load_flow_model(MODEL_NAME, device="cpu" if cpu_offloading else device).eval()
    print("Loading autoencoder...")
    ae = load_ae(MODEL_NAME, device=device).eval()

    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

        ctx = text_encoder([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        if cpu_offloading:
            text_encoder = text_encoder.cpu()
            torch.cuda.empty_cache()
            model = model.to(device)

        shape = (1, 128, height // 16, width // 16)

        def run_once(seed_val: int):
            generator = torch.Generator(device="cuda").manual_seed(seed_val)
            randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
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
            return out, elapsed

        print(f"Warmup ({warmup} run(s))...")
        for _ in range(warmup):
            run_once(seed)

        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)

        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(runs):
            out, elapsed = run_once(seed + i)
            times.append(elapsed)
            _save(out, out_dir / f"bench_klein4b_{i}.png")
            print(f"  run {i + 1}/{runs}: {elapsed * 1000:.1f} ms")

    avg = sum(times) / len(times)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(
        f"\n{width}x{height}, {num_steps} steps, guidance={guidance}\n"
        f"avg {avg * 1000:.1f} ms  (min {min(times) * 1000:.1f} ms, max {max(times) * 1000:.1f} ms)\n"
        f"peak GPU memory (resident weights + denoise/decode activations): {peak_mem:.2f} GB\n"
        f"saved images to {out_dir}/"
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
