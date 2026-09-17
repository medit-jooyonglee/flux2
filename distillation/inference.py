#!/usr/bin/env python3
"""Inference for a capacity-distilled FLUX.2 Klein Base 4B Student.

The Student is *not* guidance-distilled or step-distilled. Inference therefore
uses the Base model protocol: classifier-free guidance (negative/empty prompt +
positive prompt) and a full multi-step schedule, 50 steps by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from PIL.PngImagePlugin import PngInfo

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from student import build_student  # noqa: E402

from flux2.sampling import batched_prc_img, batched_prc_txt, denoise_cfg, get_schedule, scatter_ids  # noqa: E402
from flux2.util import load_ae, load_text_encoder  # noqa: E402

MODEL_NAME = "flux.2-klein-base-4b"
STEP_PATTERN = re.compile(r"student_step(\d+)\.pt$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def dimension(value: str) -> int:
    parsed = positive_int(value)
    if parsed % 16:
        raise argparse.ArgumentTypeError("must be divisible by 16")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="student_step<N>.pt/student_final.pt, or a directory containing checkpoints.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="CFG unconditional text. Empty prompt is the Base-model default.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/distillation"))
    parser.add_argument("--num-images", type=positive_int, default=1)
    parser.add_argument("--width", type=dimension, default=1024)
    parser.add_argument("--height", type=dimension, default=1024)
    parser.add_argument("--steps", type=positive_int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None, help="Base seed; omitted means random.")
    parser.add_argument("--device", default="cuda", help="CUDA_VISIBLE_DEVICES is set by run_inferene.sh.")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--filename-prefix", default="student")
    return parser.parse_args()


def resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = [item for item in path.glob("student_step*.pt") if STEP_PATTERN.search(item.name)]
    if candidates:
        return max(candidates, key=lambda item: int(STEP_PATTERN.search(item.name).group(1)))
    final = path / "student_final.pt"
    if final.is_file():
        return final.resolve()
    raise FileNotFoundError(f"No student_step<N>.pt or student_final.pt found in: {path}")


def load_student_checkpoint(path: Path, device: torch.device, dtype: torch.dtype):
    # mmap prevents a legacy 12.6 GB model+optimizer checkpoint from eagerly reading
    # unused AdamW moments into RAM. Only the model mapping is retained below.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
        step = payload.get("step")
        had_optimizer = "optimizer" in payload
    else:
        state_dict = payload
        match = STEP_PATTERN.search(path.name)
        step = int(match.group(1)) if match else None
        had_optimizer = False

    print(
        f"Loading Student checkpoint: {path}"
        f" (step={step if step is not None else 'unknown'}, optimizer_ignored={had_optimizer})",
        flush=True,
    )
    student = build_student(device, teacher=None)
    incompatible = student.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    student.to(dtype=dtype).eval().requires_grad_(False)
    del state_dict, payload
    gc.collect()
    return student, step


@torch.no_grad()
def encode_cfg_prompts(prompt: str, negative_prompt: str, device: torch.device, dtype: torch.dtype):
    print("Loading Qwen text encoder...", flush=True)
    text_encoder = load_text_encoder(MODEL_NAME, device=device).eval()
    # denoise_cfg expects [unconditional/negative, conditional] in that order.
    embeddings = text_encoder([negative_prompt, prompt]).to(device=device, dtype=dtype)
    ctx, ctx_ids = batched_prc_txt(embeddings)
    text_encoder.cpu()
    del text_encoder, embeddings
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return ctx, ctx_ids


def save_png(image: Image.Image, path: Path, metadata: dict) -> None:
    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    image.save(path, pnginfo=png_info)


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Encode text first and release the 4B text encoder before loading the Student.
    ctx, ctx_ids = encode_cfg_prompts(args.prompt, args.negative_prompt, device, dtype)
    student, checkpoint_step = load_student_checkpoint(checkpoint, device, dtype)
    print("Loading VAE...", flush=True)
    ae = load_ae(MODEL_NAME, device=device).eval()

    base_seed = args.seed if args.seed is not None else secrets.randbelow(2**63 - args.num_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    records = []

    latent_shape = (1, 128, args.height // 16, args.width // 16)
    for index in range(args.num_images):
        seed = base_seed + index
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latent_shape, generator=generator, dtype=dtype, device=device)
        x, x_ids = batched_prc_img(noise)
        timesteps = get_schedule(args.steps, x.shape[1])

        print(
            f"Generating {index + 1}/{args.num_images}: seed={seed}, "
            f"steps={args.steps}, guidance={args.guidance}",
            flush=True,
        )
        x = denoise_cfg(
            student,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=args.guidance,
        )
        latent = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        decoded = ae.decode(latent).float().clamp(-1, 1)[0]
        pixels = (127.5 * (rearrange(decoded, "c h w -> h w c") + 1.0)).byte().cpu().numpy()
        image = Image.fromarray(pixels)

        output_path = args.output_dir / f"{args.filename_prefix}_{run_id}_{index + 1:04d}_seed{seed}.png"
        metadata = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_step,
            "steps": args.steps,
            "guidance": args.guidance,
            "width": args.width,
            "height": args.height,
        }
        save_png(image, output_path, metadata)
        records.append({"file": str(output_path), **metadata})
        print(f"Saved: {output_path}", flush=True)

    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "base_seed": base_seed,
        "images": records,
    }
    manifest_path = args.output_dir / f"manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
