#!/usr/bin/env python3
"""Run FLUX.2 Klein inference after permanently fusing a trained LoRA in memory."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DISTILLED_MODEL = "black-forest-labs/FLUX.2-klein-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse a FLUX.2 Klein LoRA and run text-to-image or image-editing inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("t2i", "edit"), required=True)
    parser.add_argument("--model", default=DEFAULT_DISTILLED_MODEL)
    parser.add_argument("--lora", required=True, help="LoRA output directory, file, or Hub model ID.")
    parser.add_argument("--weight-name", help="Specific LoRA filename when --lora is a directory or Hub ID.")
    parser.add_argument("--adapter-name", default="trained_lora")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Strength applied while fusing the LoRA.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--input-image", type=Path, help="Condition image; required in edit mode.")
    parser.add_argument("--output", type=Path, default=Path("output/lora_result.png"))
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4, help="Use about 50 for a Base-model validation run.")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use Accelerate model CPU offload when the whole pipeline does not fit in VRAM.",
    )
    parser.add_argument(
        "--save-fused-model",
        type=Path,
        help="Optionally save the full fused pipeline. This is much larger than the LoRA adapter.",
    )
    args = parser.parse_args()

    if args.mode == "edit" and args.input_image is None:
        parser.error("--input-image is required when --mode=edit")
    if args.mode == "t2i" and args.input_image is not None:
        parser.error("--input-image is only valid when --mode=edit")
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be specified together")
    if args.num_images < 1:
        parser.error("--num-images must be at least 1")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.height is not None and (args.height % 16 or args.width % 16):
        parser.error("--height and --width must be multiples of 16")
    return args


def _output_path(base: Path, index: int, total: int) -> Path:
    if total == 1:
        return base
    suffix = base.suffix or ".png"
    return base.with_name(f"{base.stem}_{index:03d}{suffix}")


def main() -> None:
    args = parse_args()

    try:
        import torch
        from diffusers import Flux2KleinPipeline
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Missing LoRA inference dependencies. Build the environment described in training/README.md first."
        ) from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but torch.cuda.is_available() is false")

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    print(f"Loading model: {args.model}", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtypes[args.dtype])

    load_kwargs = {"adapter_name": args.adapter_name}
    if args.weight_name:
        load_kwargs["weight_name"] = args.weight_name
    print(f"Loading LoRA: {args.lora}", flush=True)
    pipe.load_lora_weights(args.lora, **load_kwargs)

    # Fusion scale must be supplied here. Passing an attention scale after this
    # point does not change weights that have already been merged.
    print(f"Fusing LoRA into transformer (scale={args.lora_scale:g})", flush=True)
    pipe.fuse_lora(
        components=["transformer"],
        lora_scale=args.lora_scale,
        safe_fusing=True,
        adapter_names=[args.adapter_name],
    )
    # Drop PEFT adapter modules after merge. Inference below therefore uses only
    # the fused transformer weights and cannot accidentally apply the LoRA twice.
    pipe.unload_lora_weights()

    if args.save_fused_model:
        args.save_fused_model.mkdir(parents=True, exist_ok=True)
        print(f"Saving full fused pipeline: {args.save_fused_model}", flush=True)
        pipe.save_pretrained(args.save_fused_model, safe_serialization=True)

    if args.cpu_offload:
        if not args.device.startswith("cuda"):
            raise ValueError("--cpu-offload requires a CUDA device")
        pipe.enable_model_cpu_offload(device=args.device)
        generator_device = args.device
    else:
        pipe.to(args.device)
        generator_device = args.device

    call_kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.num_images,
        "generator": torch.Generator(device=generator_device).manual_seed(args.seed),
    }
    if args.height is not None:
        call_kwargs.update(height=args.height, width=args.width)
    if args.mode == "edit":
        input_path = args.input_image.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        call_kwargs["image"] = Image.open(input_path).convert("RGB")

    with torch.inference_mode():
        images = pipe(**call_kwargs).images

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        output_path = _output_path(args.output, index, len(images))
        image.save(output_path)
        print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
