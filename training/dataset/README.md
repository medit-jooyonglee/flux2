# Building the smile-design LoRA dataset

Goal: teach the model to take a **generic face photo** and apply a **specified teeth/smile
style** to it, while leaving everything else (identity, background, lighting) unchanged.
Training format is the same (condition_image, target_image, instruction) triplet used by
`train_dreambooth_lora_flux2_klein_img2img.py` (see `../launch_lora_train.sh`).

## 1. Source generic face photos

- Prefer photos you have clear usage rights to. If pulling from a public face dataset
  (e.g. FFHQ), check its license fits your use (FFHQ is Nvidia-licensed for research use
  by default -- not automatically cleared for a commercial product; confirm with your
  legal/compliance contact before using it beyond internal experimentation).
- Any real patient/clinical photos need consent + de-identification per your org's data
  policy before they go anywhere near this pipeline. Keep this dataset directory out of
  git (a top-level `.gitignore` entry for `training/dataset/smile_design/images/` and
  `metadata.jsonl` is recommended once real photos are involved).
- Aim for diversity: skin tone, age, gender, lighting, camera angle, existing dental
  condition (crowding, discoloration, gaps) -- the LoRA will only generalize to the
  variation it sees.

## 2. Define your style vocabulary

Write out the distinct "teeth style" instructions you want the model to learn as
short, literal descriptions of the *transformation* (not just an end-state adjective) --
e.g.:

```
make the smile naturally bright white and evenly aligned
close the small gap between the front teeth, keep everything else unchanged
subtle veneer-style smile, slightly larger and more uniform teeth
correct the visible overbite, natural-looking result
```

Keep instructions consistent in phrasing across the dataset -- the model learns the
mapping from instruction phrasing to visual effect, so inconsistent wording for the same
effect just adds noise.

## 3. Bootstrap candidate targets, then curate

Real before/after clinical pairs are the highest-quality signal but low volume. To reach
useful LoRA training volume (see `../README.md` for the size recommendation), bootstrap
synthetic candidates using the *current* (un-fine-tuned) FLUX.2 klein 4B model against
your generic faces + style vocabulary, then keep only the ones that hold up:

```bash
cd training/dataset
PYTHONPATH=../../src python3 build_dataset.py bootstrap \
  --faces_dir /path/to/generic_faces \
  --styles_file styles.txt \
  --staging_dir ./staging
```

This calls `scripts/bench_klein4b.infer()` with each face as the reference/condition
image and each style line as the prompt, and writes one candidate PNG per (face, style)
pair into `./staging/`, plus `staging/candidates.jsonl` recording which face/style
produced which file.

**Review every candidate before it enters training.** At minimum, reject any candidate
where:
- Identity outside the mouth region visibly drifted (see `../eval/evaluate.py`'s
  `identity_preservation` check -- run it on candidates too, not just final results, to
  pre-filter automatically before your own visual pass).
- Teeth anatomy is wrong (extra/missing teeth, impossible shapes, asymmetric artifacts).
- The instruction wasn't actually followed.

Promote each approved candidate into the training set:

```bash
python3 build_dataset.py add \
  --condition /path/to/generic_faces/face001.jpg \
  --target ./staging/face001__12345678.png \
  --instruction "close the small gap between the front teeth, keep everything else unchanged"
```

This copies both images into `smile_design/images/` and appends a row to
`smile_design/metadata.jsonl`, which `smile_design/smile_design.py` (the HF `datasets`
loading script) reads at training time.

## 4. Mix in real data where you have it

If/when real before/after pairs become available (clinical outcomes, real smile-design
mockups your team has produced), add them the same way via `build_dataset.py add` --
real pairs should be weighted more heavily in review priority since they anchor the
LoRA to actually-achievable, realistic results rather than purely what the base model
already imagines plausible.

## 5. Sanity-check before training

```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('./smile_design')['train']
print(len(ds), 'examples')
print(ds[0]['instruction'])
ds[0]['condition_image'].show()  # or .save('check.png') if headless
"
```
