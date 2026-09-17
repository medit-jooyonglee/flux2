"""Offline preprocessing for Phase 1 capacity distillation (docs/flux2_distillation_05b_1b.md
section 29/32): encode each (image, prompt) pair once through the frozen VAE and Qwen3-4B
text encoder, and cache the results, so `train_distill.py` never has to run either during
training -- only the (much smaller) Teacher/Student transformer runs per training step.

Reads: <data_dir>/*.png + matching *.json (produced by scripts/dental_prompts.py's CLI:
  {"prompt": "...", ...other attrs}). Any dataset in this exact (image, {"prompt": ...})
shape works, not just dental_prompts.py output.

Writes one directory per sample under <output_dir>:
  <output_dir>/000000/latent.pt      # AE-encoded image latent, (128, H/16, W/16) bf16
  <output_dir>/000000/embedding.pt   # Qwen3-4B text embedding, (512, 7680) bf16
  <output_dir>/000000/prompt.txt     # original prompt, for debugging/provenance

Usage:
  PYTHONPATH=../src python prepare_cache.py \
    --data_dir=/data1/jooyonglee/smiledesign \
    --output_dir=/data1/jooyonglee/smiledesign_cache \
    --limit=30          # smoke test: only cache the first 30 samples
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flux2.sampling import default_images_prep  # noqa: E402
from flux2.util import load_ae, load_text_encoder  # noqa: E402

MODEL_NAME = "flux.2-klein-base-4b"  # Teacher per the distillation doc's recommendation


def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img_path in sorted(data_dir.glob("*.png")):
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            pairs.append((img_path, json_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only cache the first N samples (smoke test).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(data_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"No (image, json) pairs found in {data_dir}")

    def is_cached(i: int) -> bool:
        sample_dir = output_dir / f"{i:06d}"
        return (sample_dir / "latent.pt").exists() and (sample_dir / "embedding.pt").exists()

    num_already_cached = sum(is_cached(i) for i in range(len(pairs)))
    if num_already_cached == len(pairs):
        print(f"All {len(pairs)} samples already cached in {output_dir} -- nothing to do, skipping model load.")
        return
    print(
        f"Caching {len(pairs)} samples from {data_dir} -> {output_dir} "
        f"({num_already_cached} already done, {len(pairs) - num_already_cached} remaining)"
    )

    device = torch.device(args.device)
    print(f"Loading frozen VAE + Qwen3-4B text encoder ({MODEL_NAME}) onto {device}...")
    ae = load_ae(MODEL_NAME, device=device).eval()
    text_encoder = load_text_encoder(MODEL_NAME, device=device).eval()

    skipped = 0
    with torch.no_grad():
        for i, (img_path, json_path) in enumerate(pairs):
            sample_dir = output_dir / f"{i:06d}"
            latent_path = sample_dir / "latent.pt"
            embedding_path = sample_dir / "embedding.pt"
            if latent_path.exists() and embedding_path.exists():
                continue  # resume-friendly: skip already-cached samples

            prompt = json.loads(json_path.read_text(encoding="utf-8"))["prompt"]
            img = Image.open(img_path).convert("RGB")
            if img.width % 16 != 0 or img.height % 16 != 0:
                skipped += 1
                print(f"  ! skipping {img_path.name}: {img.size} not a multiple of 16")
                continue

            img_tensor = default_images_prep(img).unsqueeze(0).to(device)  # (1, 3, H, W) in [-1, 1]
            latent = ae.encode(img_tensor)[0].to(torch.bfloat16).cpu()  # (128, H/16, W/16)

            embedding = text_encoder([prompt])[0].to(torch.bfloat16).cpu()  # (512, 7680)

            sample_dir.mkdir(parents=True, exist_ok=True)
            torch.save(latent, latent_path)
            torch.save(embedding, embedding_path)
            (sample_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
                print(f"  [{i + 1}/{len(pairs)}] cached (latent {tuple(latent.shape)}, embedding {tuple(embedding.shape)})")

    print(f"Done. {len(pairs) - skipped} samples cached to {output_dir} ({skipped} skipped).")


if __name__ == "__main__":
    main()
