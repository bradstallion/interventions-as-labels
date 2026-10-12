"""Single entry point for loading a hooked model.

Two TransformerLens paths, selected by the model config:

- default: ``HookedTransformer.from_pretrained`` — TL's own re-implementation
  of the architecture (weights converted into TL's modules);
- ``bridge: true``: ``TransformerBridge.boot_transformers`` — TL 3.x wrapper
  that keeps the HuggingFace model and forward pass intact and only attaches
  hook points. Needed when the legacy port is broken: OLMo-3 loaded with a
  zero-filled W_K and a mis-set attention window, producing last-position
  activations identical across prompts (context-sensitivity canary,
  2026-07-06); the same model behind the bridge passes the canary because
  the computation is HF's own.

Both objects expose the interface the experiments rely on: run_with_cache,
hooks(fwd_hooks=...), to_tokens, cfg.n_layers, tokenizer, __call__ -> logits.
The downstream canaries (token replay, positional, context-sensitivity)
remain the arbiter that a load — either path — is actually sound.
"""

from __future__ import annotations

from typing import Any

import torch


def load_hooked(
    name: str,
    device: Any,
    dtype: torch.dtype,
    *,
    bridge: bool = False,
    hf_model: Any = None,
    no_processing: bool = False,
):
    """no_processing: skip TL's weight preprocessing (fold_ln,
    center_writing_weights, ...). The preprocessing preserves the computation
    but CHANGES the residual-stream values — anything trained on raw HF
    activations (Gemma Scope SAEs) reads garbage in the processed basis
    (M6-SAE T0 gate: FVU 8.65 processed vs healthy raw). Directions fit and
    used within one run are basis-consistent either way."""
    if bridge:
        from transformer_lens.model_bridge import TransformerBridge

        if hf_model is None:
            return TransformerBridge.boot_transformers(
                name, device=device, dtype=dtype)
        # boot_transformers moves a model it loads itself, but NOT one handed
        # to it: a CPU-resident hf_model silently runs the whole eval on CPU
        # (observed 2026-07-06: edit_eval at 2600% CPU, GPU empty). Move it
        # ourselves and verify the landing.
        hf_model = hf_model.to(device=device, dtype=dtype)
        model = TransformerBridge.boot_transformers(
            name, device=device, dtype=dtype, hf_model=hf_model)
        got = next(model.parameters()).device.type
        want = torch.device(device).type
        if got != want:
            raise AssertionError(
                f"bridge model landed on {got}, requested {want} — silent "
                "CPU fallback would invalidate runtimes and numbers")
        return model
    import transformer_lens as tl

    kwargs = {"hf_model": hf_model} if hf_model is not None else {}
    loader = (tl.HookedTransformer.from_pretrained_no_processing
              if no_processing else tl.HookedTransformer.from_pretrained)
    return loader(name, device=device, dtype=dtype, **kwargs)
