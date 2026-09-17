"""Reads the cache directory built by prepare_cache.py.

All samples in a given cache dir share the same latent/embedding shape (fixed 512-token
padded text embeddings, and every source image in this Phase 1 recipe is the same
1024x1024 size), so plain default collation/stacking is enough -- no bucketing/padding
logic like the LoRA image-editing trainer needs.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedLatentDataset(Dataset):
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.sample_dirs = sorted(
            d for d in self.cache_dir.iterdir() if d.is_dir() and (d / "latent.pt").exists()
        )
        if not self.sample_dirs:
            raise ValueError(f"No cached samples found in {cache_dir} -- run prepare_cache.py first.")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.sample_dirs[idx]
        prompt_path = sample_dir / "prompt.txt"
        return {
            "latent": torch.load(sample_dir / "latent.pt", weights_only=True),
            "embedding": torch.load(sample_dir / "embedding.pt", weights_only=True),
            "prompt": prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "",
        }
