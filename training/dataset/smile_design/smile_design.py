"""HF `datasets` loading script for the smile-design LoRA training set.

Exists because the img2img training script (train_dreambooth_lora_flux2_klein_img2img.py)
calls `datasets.load_dataset(path, ...)` and expects `dataset["train"][image_column]` /
`dataset["train"][cond_image_column]` to already be decoded PIL images -- which only
happens if both image columns are explicitly typed as `datasets.Image()`. The generic
`imagefolder` builder only auto-decodes a single designated image column, not two, so a
custom loading script is the reliable way to get both columns decoded.

Kept fully local and offline on purpose (no Hub push) since this dataset contains
face/dental photos.

Usage: `--dataset_name=<path to this smile_design/ directory>` when launching training.
Populate metadata.jsonl (and the images/ it references) via ../build_dataset.py.
"""

import json
import os

import datasets

_METADATA_FILENAME = "metadata.jsonl"


class SmileDesignDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Condition/target/instruction triplets for smile-design LoRA fine-tuning.",
            features=datasets.Features(
                {
                    "condition_image": datasets.Image(),
                    "target_image": datasets.Image(),
                    "instruction": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        root = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(root, _METADATA_FILENAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"{metadata_path} not found -- run ../build_dataset.py first to populate this dataset directory."
            )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"root": root, "metadata_path": metadata_path},
            )
        ]

    def _generate_examples(self, root: str, metadata_path: str):
        with open(metadata_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield idx, {
                    "condition_image": os.path.join(root, row["condition_image"]),
                    "target_image": os.path.join(root, row["target_image"]),
                    "instruction": row["instruction"],
                }
