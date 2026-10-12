"""On-disk activation store.

Layout (one directory per capture run):

    <root>/
      meta.json                       # model, seed, git sha, config hash
      index.json                      # {site: {prompt_idx: shape}}
      <site_sanitized>.safetensors    # one file per site; key "p{idx}" per prompt

Design constraints: activations are cached
once and are the expensive artifact; everything derived (probes, directions,
metrics) is recomputed from here. Prompts may have different lengths, so
tensors are stored per-prompt (no padding is ever stored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

_META = "meta.json"
_INDEX = "index.json"


def _sanitize(site: str) -> str:
    """Hook names contain dots; keep filenames flat and reversible."""
    return site.replace(".", "__")


class ActivationStore:
    """Write-once, read-many store of per-prompt activation tensors."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        index_path = self.root / _INDEX
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found — use ActivationStore.write() to create a store"
            )
        self.index: dict[str, dict[str, list[int]]] = json.loads(index_path.read_text())
        self.meta: dict[str, Any] = json.loads((self.root / _META).read_text())
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    # ---------- creation ----------

    @classmethod
    def write(
        cls,
        root: Path | str,
        activations: dict[str, dict[int, torch.Tensor]],
        meta: dict[str, Any],
    ) -> "ActivationStore":
        """Persist {site: {prompt_idx: tensor}} + metadata and return the store.

        Refuses to overwrite an existing store: activation runs are immutable;
        a new run gets a new directory.
        """
        root = Path(root)
        if (root / _INDEX).exists():
            raise FileExistsError(f"store already exists at {root}")
        root.mkdir(parents=True, exist_ok=True)

        index: dict[str, dict[str, list[int]]] = {}
        for site, per_prompt in activations.items():
            tensors = {f"p{idx}": t.contiguous() for idx, t in per_prompt.items()}
            save_file(tensors, str(root / f"{_sanitize(site)}.safetensors"))
            index[site] = {str(idx): list(t.shape) for idx, t in per_prompt.items()}

        (root / _INDEX).write_text(json.dumps(index, indent=1))
        (root / _META).write_text(json.dumps(meta, indent=1, default=str))
        return cls(root)

    # ---------- access ----------

    @property
    def sites(self) -> list[str]:
        return sorted(self.index.keys())

    def prompt_indices(self, site: str) -> list[int]:
        return sorted(int(k) for k in self.index[site])

    def get(self, prompt_idx: int, site: str) -> torch.Tensor:
        if site not in self._cache:
            self._cache[site] = load_file(str(self.root / f"{_sanitize(site)}.safetensors"))
        return self._cache[site][f"p{prompt_idx}"]

    def shape(self, prompt_idx: int, site: str) -> list[int]:
        return self.index[site][str(prompt_idx)]
