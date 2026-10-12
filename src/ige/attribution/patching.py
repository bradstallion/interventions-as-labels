"""Activation patching between matched twins.

Effect metric for τ1: Δ(logit["ASK"] − logit["ACTION"]) at the final prompt
position — the model's next-token preference between asking and acting,
measured in one forward pass, no generation. Patching the unambiguous twin's
activation into the ambiguous run (and vice versa) at a causal site should
move this preference; random sites should not (control).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def ask_action_token_ids(tokenizer) -> tuple[int, int]:
    """First token of 'ASK' / 'ACTION' as reply openers (no leading space:
    replies start at line start after the generation prompt)."""
    ask = tokenizer("ASK", add_special_tokens=False)["input_ids"][0]
    action = tokenizer("ACTION", add_special_tokens=False)["input_ids"][0]
    if ask == action:
        raise ValueError("ASK/ACTION share the first token — metric undefined")
    return ask, action


@torch.no_grad()
def ask_action_logit_diff(model, tokens: torch.Tensor, ids: tuple[int, int]) -> float:
    """logit(ASK) − logit(ACTION) at the last position."""
    logits = model(tokens)[0, -1]
    return float(logits[ids[0]] - logits[ids[1]])


@dataclass
class PatchEffect:
    site: str
    baseline: float   # logit diff of the destination run, unpatched
    patched: float    # logit diff after donor activation is patched in
    delta: float      # patched - baseline

    def to_dict(self) -> dict:
        return self.__dict__


@torch.no_grad()
def patch_last_position(
    model,
    dst_tokens: torch.Tensor,
    src_tokens: torch.Tensor,
    site: str,
    ids: tuple[int, int],
) -> PatchEffect:
    """Replace `site` activation at the LAST position of the destination run
    with the donor's last-position activation; report the effect on the
    ASK−ACTION preference. Twins may differ in length: last position each.
    """
    _, src_cache = model.run_with_cache(src_tokens, names_filter=lambda n: n == site)
    donor = src_cache[site][0, -1]

    baseline = ask_action_logit_diff(model, dst_tokens, ids)

    def hook(value, hook):  # value: [batch, pos, d]
        value[0, -1] = donor.to(value.dtype)
        return value

    with model.hooks(fwd_hooks=[(site, hook)]):
        logits = model(dst_tokens)[0, -1]
    patched = float(logits[ids[0]] - logits[ids[1]])
    return PatchEffect(site=site, baseline=baseline, patched=patched,
                       delta=patched - baseline)
