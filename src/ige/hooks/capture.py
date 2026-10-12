"""Activation capture behind a single API.

Backend: TransformerLens (primary). The point of this module is that nothing
outside ige.hooks imports transformer_lens directly, so an nnsight backend can
be added without touching callers.

Prompts are processed one at a time: variable lengths without padding, and
peak memory stays flat — throughput matters less than correctness here, and
batching can be added behind the same API later.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from ige.hooks.store import ActivationStore
from ige.utils.env import run_metadata


@torch.no_grad()
def capture(
    model,  # transformer_lens.HookedTransformer
    prompts: list[str],
    sites: list[str],
    out_dir: Path | str,
    *,
    store_dtype: torch.dtype = torch.float16,
    extra_meta: dict[str, Any] | None = None,
) -> ActivationStore:
    """Run the model on each prompt, persist activations at `sites`.

    Returns the on-disk ActivationStore. Tensors are stored per prompt with
    shape [seq, ...] (batch dim stripped), cast to `store_dtype` on CPU.
    """
    site_set = set(sites)
    unknown = site_set - set(model.hook_dict.keys())
    if unknown:
        raise ValueError(f"unknown hook sites for this model: {sorted(unknown)}")

    activations: dict[str, dict[int, torch.Tensor]] = {s: {} for s in sites}
    for idx, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in site_set)
        for site in sites:
            activations[site][idx] = cache[site][0].to("cpu", dtype=store_dtype)

    meta = run_metadata(
        {"model": model.cfg.model_name, "sites": sites, "n_prompts": len(prompts)}
    )
    meta.update(
        {
            "model": model.cfg.model_name,
            "sites": sites,
            "n_prompts": len(prompts),
            "store_dtype": str(store_dtype),
            "prompt_digests": [hashlib.sha256(p.encode()).hexdigest()[:8] for p in prompts],
        }
    )
    if extra_meta:
        meta.update(extra_meta)
    return ActivationStore.write(out_dir, activations, meta)
