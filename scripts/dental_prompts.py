"""Random prompt generator for diverse face/teeth images.

Combines ethnicity/age/gender/shot-angle/expression/dental-condition keywords into
prompts for klein 4B, so you get varied synthetic faces -- not just frontal close-ups --
to stress-test dental imaging/detection pipelines, or to seed the generic-face pool for
the smile-design LoRA data pipeline (training/dataset/README.md's `bootstrap` step: run
this first to build a diverse `--faces_dir`, then apply teeth-style edit instructions on
top of those faces). Each generated prompt comes with the attribute combination used to
build it, so you can save it alongside the image as a label for later use.

Usage as a library:
  from dental_prompts import random_dental_prompt, random_dental_prompts

  prompt, attrs = random_dental_prompt()
  batch = random_dental_prompts(20, seed=0)  # reproducible batch

Usage as a script (calls scripts/server.py's /generate endpoint in a loop):
  python scripts/dental_prompts.py --count=20 --server=http://127.0.0.1:8000 --output_dir=output/dental
"""

import random

ETHNICITIES = [
    "East Asian",
    "Southeast Asian",
    "South Asian",
    "Black African",
    "Caucasian European",
    "Hispanic Latino",
    "Middle Eastern",
    "Native American",
]

AGE_GROUPS = [
    "child around 10 years old",
    "teenager",
    "young adult in their 20s",
    "middle-aged adult in their 40s",
    "elderly person in their 70s",
]

GENDERS = ["male", "female"]

# Shot framing + camera angle, combined into one natural phrase per entry (rather than
# crossing two separate lists) so every combination still reads coherently in the prompt.
SHOTS = [
    "a frontal close-up portrait photo",
    "a three-quarter angle close-up portrait photo, head turned slightly to the left",
    "a three-quarter angle close-up portrait photo, head turned slightly to the right",
    "a left profile portrait photo",
    "a right profile portrait photo",
    "a close-up portrait photo taken from slightly above eye level, looking down at the camera",
    "a close-up portrait photo taken from slightly below eye level, looking up at the camera",
    "an extreme close-up photo focused tightly on the mouth and teeth",
    "a casual phone selfie photo, arm's-length framing, slight wide-angle lens distortion",
    "a professional clinical dental photograph with cheek retractors, frontal intraoral view",
    "a professional clinical dental photograph with cheek retractors, lateral intraoral view",
    "a candid photo mid-conversation, head turned and mouth caught mid-motion",
]

EXPRESSIONS = [
    "smiling broadly showing teeth",
    "laughing with mouth wide open showing teeth",
    "a natural smile showing upper teeth",
]

DENTAL_CONDITIONS = [
    "perfectly healthy straight white teeth",
    "yellowed and stained teeth",
    "crooked and misaligned teeth",
    "teeth with a visible gap between the front teeth (diastema)",
    "crowded and overlapping teeth",
    "a chipped front tooth",
    "wearing metal dental braces",
    "a visible gold or ceramic dental crown",
    "teeth with visible plaque buildup",
    "red and swollen gums (gingivitis)",
    "missing a front tooth",
    "a noticeable overbite",
    "a noticeable underbite",
    "wearing dentures",
]

PROMPT_TEMPLATE = (
    "{shot} of a {age_group} {ethnicity} {gender}, "
    "{expression}, {dental_condition}, natural lighting, high detail, photorealistic"
)


def random_dental_prompt(rng: random.Random | None = None) -> tuple[str, dict]:
    """Return (prompt, attrs) for one random combination. Pass an rng for reproducibility."""
    rng = rng or random

    attrs = {
        "shot": rng.choice(SHOTS),
        "age_group": rng.choice(AGE_GROUPS),
        "ethnicity": rng.choice(ETHNICITIES),
        "gender": rng.choice(GENDERS),
        "expression": rng.choice(EXPRESSIONS),
        "dental_condition": rng.choice(DENTAL_CONDITIONS),
    }
    prompt = PROMPT_TEMPLATE.format(**attrs)
    return prompt, attrs


def random_dental_prompts(n: int, seed: int | None = None) -> list[tuple[str, dict]]:
    """Return n (prompt, attrs) pairs. Same seed -> same batch, for reproducible test sets."""
    rng = random.Random(seed)
    return [random_dental_prompt(rng) for _ in range(n)]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    import requests

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--server", default="http://10.100.1.45:8602")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output_dir", default="output/dentiform")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (prompt, attrs) in enumerate(random_dental_prompts(args.count, seed=args.seed)):
        headers = {
            "X-Prompt": prompt,
            "X-Width": str(args.width),
            "X-Height": str(args.height),
            "X-Seed": str(i),
        }
        resp = requests.post(f"{args.server}/generate", headers=headers, timeout=300)
        if resp.status_code != 200:
            print(f"[{i}] FAILED {resp.status_code}: {resp.text}")
            continue

        img_path = out_dir / f"dental_{i:04d}.png"
        label_path = out_dir / f"dental_{i:04d}.json"
        img_path.write_bytes(resp.content)
        label_path.write_text(json.dumps({"prompt": prompt, **attrs}, ensure_ascii=False, indent=2))
        print(f"[{i + 1}/{args.count}] {attrs['dental_condition']!r} -> {img_path}")
