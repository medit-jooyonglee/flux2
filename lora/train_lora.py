#!/usr/bin/env python3
"""FLUX.2 Klein 4B LoRA training entry point.

This script validates and loads a local text-to-image or image-editing dataset,
then delegates the actual PEFT/Accelerate training loop to the pinned Diffusers
trainers already vendored in ``training/external``.

The wrapper intentionally owns the local dataset loading. Hugging Face's generic
``imagefolder`` loader only decodes one image column reliably, whereas edit
training needs both the condition and target columns to be decoded as PIL images.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINERS = {
    "t2i": REPO_ROOT / "training" / "external" / "train_dreambooth_lora_flux2_klein.py",
    "edit": REPO_ROOT / "training" / "external" / "train_dreambooth_lora_flux2_klein_img2img.py",
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
RESERVED_TRAINER_OPTIONS = {
    "--dataset_name",
    "--dataset_config_name",
    "--instance_data_dir",
    "--image_column",
    "--cond_image_column",
    "--caption_column",
}


@dataclass(frozen=True)
class T2ISample:
    image: Path
    caption: str


@dataclass(frozen=True)
class EditSample:
    condition_image: Path
    target_image: Path
    instruction: str


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train a FLUX.2 Klein transformer LoRA for T2I or image editing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=("t2i", "edit"), required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Local dataset root. See lora/README.md for the supported layouts.",
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the pinned Diffusers trainer.",
    )
    args = parser.parse_args()

    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args.pop(0)
    return args, trainer_args


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield line_number, row


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _resolve_image(root: Path, value: Any, source: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: image path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{source}: image not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"{source}: unsupported image extension: {path.suffix}")
    return path


def _clean_text(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: caption/instruction must be a non-empty string")
    return value.strip()


def _find_matching_file(root: Path, relative_image: Path, suffixes: tuple[str, ...]) -> Path:
    same_suffix = root / relative_image
    if same_suffix.is_file() and same_suffix.suffix.lower() in suffixes:
        return same_suffix
    for suffix in suffixes:
        candidate = (root / relative_image).with_suffix(suffix)
        if candidate.is_file():
            return candidate
    expected = (root / relative_image).with_suffix(suffixes[0])
    raise FileNotFoundError(f"matching file not found for {relative_image}; expected near {expected}")


def load_t2i_samples(root: Path) -> list[T2ISample]:
    metadata = root / "metadata.jsonl"
    samples: list[T2ISample] = []

    if metadata.is_file():
        for line_number, row in _read_jsonl(metadata):
            source = f"{metadata}:{line_number}"
            image = _resolve_image(root, _first_value(row, ("image", "file_name", "target_image")), source)
            caption = _clean_text(_first_value(row, ("caption", "text", "prompt", "instruction")), source)
            samples.append(T2ISample(image=image, caption=caption))
    else:
        images = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image in images:
            caption_file = image.with_suffix(".txt")
            if not caption_file.is_file():
                relative = image.relative_to(root)
                caption_file = _find_matching_file(root / "caption", relative, (".txt",))
            caption = _clean_text(caption_file.read_text(encoding="utf-8"), str(caption_file))
            samples.append(T2ISample(image=image.resolve(), caption=caption))

    if not samples:
        raise ValueError(f"No T2I samples found in {root}")
    return samples


def load_edit_samples(root: Path) -> list[EditSample]:
    metadata = root / "metadata.jsonl"
    samples: list[EditSample] = []

    if metadata.is_file():
        for line_number, row in _read_jsonl(metadata):
            source = f"{metadata}:{line_number}"
            condition = _resolve_image(
                root,
                _first_value(row, ("condition_image", "input_image", "input")),
                source,
            )
            target = _resolve_image(root, _first_value(row, ("target_image", "target")), source)
            instruction = _clean_text(
                _first_value(row, ("instruction", "caption", "text", "prompt")),
                source,
            )
            samples.append(EditSample(condition, target, instruction))
    else:
        input_root = root / "input"
        if not input_root.is_dir():
            input_root = root / "condition"
        target_root = root / "target"
        caption_root = root / "caption"
        if not input_root.is_dir() or not target_root.is_dir() or not caption_root.is_dir():
            raise FileNotFoundError(
                "Edit data requires metadata.jsonl or input(or condition)/, target/, and caption/ directories"
            )

        inputs = sorted(
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for condition in inputs:
            relative = condition.relative_to(input_root)
            target = _find_matching_file(target_root, relative, IMAGE_SUFFIXES)
            caption_file = _find_matching_file(caption_root, relative, (".txt",))
            instruction = _clean_text(caption_file.read_text(encoding="utf-8"), str(caption_file))
            samples.append(EditSample(condition.resolve(), target.resolve(), instruction))

    if not samples:
        raise ValueError(f"No image-editing samples found in {root}")
    return samples


def _option_was_passed(arguments: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def _reject_reserved_options(arguments: list[str]) -> None:
    conflicts = sorted(option for option in RESERVED_TRAINER_OPTIONS if _option_was_passed(arguments, option))
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"These options are managed by train_lora.py and must not be passed manually: {joined}")


def _install_dataset_loader(task: str, dataset_dir: Path, samples: list[T2ISample] | list[EditSample]) -> None:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'datasets'. Install the LoRA training environment first.") from exc

    if task == "t2i":
        t2i_samples = samples
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "caption": datasets.Value("string"),
            }
        )
        train_dataset = datasets.Dataset.from_dict(
            {
                "image": [str(sample.image) for sample in t2i_samples],
                "caption": [sample.caption for sample in t2i_samples],
            },
            features=features,
        )
    else:
        edit_samples = samples
        features = datasets.Features(
            {
                "condition_image": datasets.Image(),
                "target_image": datasets.Image(),
                "instruction": datasets.Value("string"),
            }
        )
        train_dataset = datasets.Dataset.from_dict(
            {
                "condition_image": [str(sample.condition_image) for sample in edit_samples],
                "target_image": [str(sample.target_image) for sample in edit_samples],
                "instruction": [sample.instruction for sample in edit_samples],
            },
            features=features,
        )

    original_load_dataset = datasets.load_dataset
    resolved_dataset_dir = dataset_dir.resolve()

    def load_local_dataset(path: str, *args: Any, **kwargs: Any):
        try:
            requested_path = Path(path).expanduser().resolve()
        except (OSError, TypeError, ValueError):
            requested_path = None
        if requested_path == resolved_dataset_dir:
            return datasets.DatasetDict({"train": train_dataset})
        return original_load_dataset(path, *args, **kwargs)

    datasets.load_dataset = load_local_dataset


def main() -> None:
    args, trainer_args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    trainer = TRAINERS[args.task]
    if not trainer.is_file():
        raise FileNotFoundError(f"Pinned Diffusers trainer not found: {trainer}")

    _reject_reserved_options(trainer_args)
    samples = load_t2i_samples(dataset_dir) if args.task == "t2i" else load_edit_samples(dataset_dir)
    _install_dataset_loader(args.task, dataset_dir, samples)

    managed_args = [f"--dataset_name={dataset_dir}"]
    if args.task == "t2i":
        managed_args.extend(("--image_column=image", "--caption_column=caption"))
        # The upstream trainer marks this argument as required even when every row
        # has its own caption. It is only used as a fallback in this configuration.
        if not _option_was_passed(trainer_args, "--instance_prompt"):
            managed_args.append("--instance_prompt= ")
    else:
        managed_args.extend(
            (
                "--image_column=target_image",
                "--cond_image_column=condition_image",
                "--caption_column=instruction",
            )
        )

    print(f"Validated {len(samples)} {args.task} samples from {dataset_dir}", flush=True)
    print(f"Starting pinned Diffusers trainer: {trainer.name}", flush=True)

    sys.argv = [str(trainer), *managed_args, *trainer_args]
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
