"""Build/extend the smile_design/ LoRA training set (see smile_design/smile_design.py).

Two workflows:

1. `bootstrap` -- synthesize CANDIDATE target images from generic face photos using this
   repo's own FLUX.2 [klein] 4B model (bench_klein4b.infer), one per (face, style prompt)
   pair. Candidates land in a staging directory for human review -- nothing is added to
   the training set automatically, since self-generated images can contain artifacts
   (extra/misshapen teeth, identity drift) that would poison training if used unchecked.

     PYTHONPATH=../../src python3 build_dataset.py bootstrap \
       --faces_dir /path/to/generic_faces \
       --styles_file styles.txt \
       --staging_dir ./staging

   styles.txt: one instruction per line, e.g.:
     make the smile naturally bright white and evenly aligned
     close the small gap between the front teeth, keep everything else unchanged
     subtle veneer-style smile, slightly larger and more uniform teeth

2. `add` -- after a human has reviewed a (condition, target) pair (either a bootstrapped
   candidate that passed review, or a real before/after photo pair) and confirmed it's
   usable, promote it into the dataset:

     python3 build_dataset.py add \
       --condition /path/to/face.jpg \
       --target ./staging/face_style2.png \
       --instruction "close the small gap between the front teeth, keep everything else unchanged"

Run `python3 build_dataset.py --help` for details.
"""

import argparse
import itertools
import json
import shutil
import sys
import uuid
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "smile_design"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_PATH = DATASET_DIR / "metadata.jsonl"


def cmd_add(args: argparse.Namespace) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    pair_id = uuid.uuid4().hex[:12]

    condition_src = Path(args.condition)
    target_src = Path(args.target)
    condition_dst = IMAGES_DIR / f"{pair_id}_condition{condition_src.suffix}"
    target_dst = IMAGES_DIR / f"{pair_id}_target{target_src.suffix}"
    shutil.copy(condition_src, condition_dst)
    shutil.copy(target_src, target_dst)

    row = {
        "condition_image": str(condition_dst.relative_to(DATASET_DIR)),
        "target_image": str(target_dst.relative_to(DATASET_DIR)),
        "instruction": args.instruction,
    }
    with open(METADATA_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Added pair {pair_id} to {METADATA_PATH}")


def cmd_bootstrap(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import bench_klein4b as bk  # noqa: E402  (repo's own inference pipeline)

    faces_dir = Path(args.faces_dir)
    faces = sorted([p for p in faces_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not faces:
        raise SystemExit(f"No images found in {faces_dir}")

    styles = [line.strip() for line in Path(args.styles_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not styles:
        raise SystemExit(f"No style prompts found in {args.styles_file}")

    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = staging_dir / "candidates.jsonl"

    print(f"Generating {len(faces)} faces x {len(styles)} styles = {len(faces) * len(styles)} candidates...")
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for face, style in itertools.product(faces, styles):
            out_path = staging_dir / f"{face.stem}__{abs(hash(style)) % 10**8}.png"
            if out_path.exists():
                continue  # resume-friendly: skip already-generated candidates
            img, _elapsed = bk.infer(
                prompt=style,
                input_images=str(face),
                match_image_size=0,  # keep output at the source face's resolution
                cpu_offloading=args.cpu_offloading,
            )
            img.save(out_path)
            manifest.write(
                json.dumps(
                    {"condition_image": str(face), "target_image": str(out_path), "instruction": style},
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"  wrote {out_path}")

    print(
        f"\nDone. Review candidates in {staging_dir}/, then promote the good ones with:\n"
        f"  python3 build_dataset.py add --condition <face> --target <chosen candidate> --instruction \"<style>\""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Promote one reviewed (condition, target, instruction) triplet into the dataset.")
    p_add.add_argument("--condition", required=True, help="Path to the input/condition image.")
    p_add.add_argument("--target", required=True, help="Path to the desired output/target image.")
    p_add.add_argument("--instruction", required=True, help="Instruction text describing the transformation.")
    p_add.set_defaults(func=cmd_add)

    p_boot = sub.add_parser(
        "bootstrap", help="Synthesize candidate targets from generic faces + style prompts for human review."
    )
    p_boot.add_argument("--faces_dir", required=True, help="Directory of generic face photos.")
    p_boot.add_argument("--styles_file", required=True, help="Text file, one teeth-style instruction per line.")
    p_boot.add_argument("--staging_dir", required=True, help="Where to write candidates for review.")
    p_boot.add_argument("--cpu_offloading", action="store_true", default=False)
    p_boot.set_defaults(func=cmd_bootstrap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
