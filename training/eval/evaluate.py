"""Evaluate a trained smile-design LoRA: generate outputs for a held-out eval set and
score them automatically, then hand off to human review for anything automatic metrics
can't judge (realism, whether the *dental* result looks clinically sane).

Run inside training/.venv (needs diffusers/peft/insightface/lpips, all installed there).

Usage:
  cd training/eval
  ../.venv/bin/python evaluate.py \
    --lora_dir ../output/smile-design-lora \
    --eval_manifest ./eval_manifest.jsonl \
    --out_dir ./results

eval_manifest.jsonl: one JSON object per line, held OUT of training data:
  {"condition_image": "/path/to/face.jpg", "instruction": "...", "target_image": "/path/to/real_target.jpg"}
`target_image` is optional -- include it only when you have a real reference result to
compare against (e.g. a held-out real before/after pair), which enables the LPIPS
target-similarity score in addition to the two reference-free metrics below.

Metrics (all automatic ones are proxies -- see the printed caveats):
  - identity_preservation: cosine similarity between ArcFace embeddings (insightface) of
    the condition and generated images. High = identity/face structure preserved.
    Caveat: it's a whole-face embedding, so a very heavy/localized teeth edit can lower
    this even when the edit is exactly what was asked for -- read it together with
    style_alignment, not in isolation.
  - style_alignment: CLIP image-text cosine similarity between the generated image and
    the instruction text (CLIPScore-style). Higher = image matches the requested style
    description better. Caveat: CLIP was not trained on dental imagery specifically, so
    treat this as a coarse signal, not ground truth.
  - target_similarity (only if target_image given): 1 - LPIPS distance between generated
    and real target image. Higher = closer to the known-good reference result.

None of these substitute for a human (ideally someone dental-trained) rating realism and
clinical plausibility -- this script also emits a manual_review.csv template for that.
"""

import argparse
import csv
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def face_embedding(app: FaceAnalysis, img: Image.Image) -> np.ndarray | None:
    faces = app.get(np.array(img.convert("RGB"))[:, :, ::-1])  # RGB -> BGR
    if not faces:
        return None
    # If more than one face is detected, use the largest (most likely the subject).
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


def identity_preservation(app: FaceAnalysis, condition: Image.Image, generated: Image.Image) -> float | None:
    e1, e2 = face_embedding(app, condition), face_embedding(app, generated)
    if e1 is None or e2 is None:
        return None  # no face detected in one of the two -- can't score, flag for manual review
    return float(np.dot(e1, e2))


def style_alignment(clip_model, clip_processor, device, generated: Image.Image, instruction: str) -> float:
    inputs = clip_processor(text=[instruction], images=[generated], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
    image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_embeds @ text_embeds.T).item())


def target_similarity(lpips_model, device, generated: Image.Image, target: Image.Image) -> float:
    def to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB").resize(target.size)).astype("float32") / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        dist = lpips_model(to_tensor(generated), to_tensor(target)).item()
    return 1.0 - dist


def make_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    from PIL import ImageDraw

    h = max(img.height for img in images)
    images = [img.resize((int(img.width * h / img.height), h)) for img in images]
    total_w = sum(img.width for img in images)
    grid = Image.new("RGB", (total_w, h + 24), "white")
    draw = ImageDraw.Draw(grid)
    x = 0
    for img, label in zip(images, labels):
        grid.paste(img, (x, 24))
        draw.text((x + 4, 4), label, fill="black")
        x += img.width
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_model", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--lora_dir", required=True, help="Trained LoRA output_dir (from launch_lora_train.sh).")
    parser.add_argument("--eval_manifest", required=True, help="JSONL of held-out eval examples (see module docstring).")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from diffusers import Flux2KleinPipeline

    print(f"Loading {args.base_model} + LoRA from {args.lora_dir} ...")
    pipe = Flux2KleinPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_dir)
    pipe.enable_model_cpu_offload()

    print("Loading eval models (insightface ArcFace, CLIP, LPIPS)...")
    face_app = load_face_app()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = [json.loads(line) for line in Path(args.eval_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(rows)} eval examples")

    results = []
    review_rows = []
    for i, row in enumerate(rows):
        condition = Image.open(row["condition_image"]).convert("RGB")
        instruction = row["instruction"]
        target = Image.open(row["target_image"]).convert("RGB") if row.get("target_image") else None

        generator = torch.Generator(device=device).manual_seed(args.seed)
        generated = pipe(
            image=condition,
            prompt=instruction,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]

        gen_path = out_dir / "generated" / f"{i:04d}.png"
        generated.save(gen_path)

        metrics = {
            "index": i,
            "condition_image": row["condition_image"],
            "instruction": instruction,
            "generated_image": str(gen_path),
            "identity_preservation": identity_preservation(face_app, condition, generated),
            "style_alignment": style_alignment(clip_model, clip_processor, device, generated, instruction),
        }
        if target is not None:
            metrics["target_similarity"] = target_similarity(lpips_model, device, generated, target)
        results.append(metrics)

        grid_imgs = [condition, generated] + ([target] if target is not None else [])
        grid_labels = ["condition", "generated"] + (["real target"] if target is not None else [])
        make_grid(grid_imgs, grid_labels).save(out_dir / "grids" / f"{i:04d}.png")

        review_rows.append(
            {
                "index": i,
                "grid_image": str(out_dir / "grids" / f"{i:04d}.png"),
                "instruction": instruction,
                "realism_1_5": "",
                "style_match_1_5": "",
                "clinically_plausible_teeth_y_n": "",
                "artifacts_present_y_n": "",
                "notes": "",
            }
        )
        print(f"[{i + 1}/{len(rows)}] identity={metrics['identity_preservation']!r} style={metrics['style_alignment']:.3f}")

    # Aggregate stats (skip Nones -- e.g. identity_preservation when no face was detected).
    def _mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n": len(results),
        "identity_preservation_mean": _mean("identity_preservation"),
        "style_alignment_mean": _mean("style_alignment"),
        "target_similarity_mean": _mean("target_similarity"),
        "faces_undetected": sum(1 for r in results if r.get("identity_preservation") is None),
    }
    (out_dir / "eval_report.json").write_text(json.dumps({"summary": summary, "per_example": results}, indent=2))

    with open(out_dir / "manual_review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_dir}/eval_report.json, {out_dir}/grids/*.png, and {out_dir}/manual_review.csv")
    print("Have a human (ideally dental-trained) reviewer fill in manual_review.csv before calling this LoRA good.")


if __name__ == "__main__":
    main()
