"""Causal test for a probe direction: set the activation's projection on the
direction to a target value (e.g. the acted-class mean) at the last position,
and measure whether the ASK/ACTION decision flips.

This is the honest follow-up to AUC-1.0 probes (audit `xxw`): under greedy
decoding late-layer decodability is expected, so the claim must be causal —
moving ONLY the probe coordinate should move the decision, and the same
procedure on a random direction should not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def project_to(h: torch.Tensor, unit_dir: torch.Tensor, target: float) -> torch.Tensor:
    """Return h with its scalar projection on unit_dir set to `target`."""
    proj = h @ unit_dir
    return h + (target - proj) * unit_dir


@dataclass
class DirectionPatchResult:
    site: str
    alpha_label: str          # "class0_mean" | "extrapolated" | ...
    n: int
    n_base_ask: int           # records preferring ASK before the patch
    n_base_act: int           # records preferring ACTION before the patch
    ask_to_action: int        # flips ASK -> ACTION (face 1 metric)
    action_to_ask: int        # flips ACTION -> ASK (face 2 metric)
    mean_delta_logitdiff: float

    @property
    def flip_rate(self) -> float:          # face-1 rate, kept for continuity
        return self.ask_to_action / max(1, self.n_base_ask)

    @property
    def induce_rate(self) -> float:        # face-2 rate
        return self.action_to_ask / max(1, self.n_base_act)

    def to_dict(self) -> dict:
        return {**self.__dict__, "flip_rate": self.flip_rate,
                "induce_rate": self.induce_rate}


@torch.no_grad()
def patch_direction_last_pos(
    model,
    tokens_list: list[torch.Tensor],
    site: str,
    unit_dir: torch.Tensor,
    target: float,
    ids: tuple[int, int],     # (ASK, ACTION) first-token ids
    alpha_label: str,
) -> DirectionPatchResult:
    """Apply project_to at `site`, last position, for every tokens tensor;
    count decision flips in BOTH directions and the mean logit-diff change."""
    ask_to_action = action_to_ask = n_ask = n_act = 0
    deltas = []
    device = tokens_list[0].device if tokens_list else "cpu"
    d = unit_dir.float().to(device)

    for tokens in tokens_list:
        base_logits = model(tokens)[0, -1]
        base_diff = float(base_logits[ids[0]] - base_logits[ids[1]])

        def hook(value, hook):
            value[0, -1] = project_to(value[0, -1].float(), d,
                                      float(target)).to(value.dtype)
            return value

        with model.hooks(fwd_hooks=[(site, hook)]):
            logits = model(tokens)[0, -1]
        diff = float(logits[ids[0]] - logits[ids[1]])
        deltas.append(diff - base_diff)
        if base_diff > 0:
            n_ask += 1
            if diff < 0:
                ask_to_action += 1
        else:
            n_act += 1
            if diff > 0:
                action_to_ask += 1

    return DirectionPatchResult(
        site=site, alpha_label=alpha_label, n=len(tokens_list),
        n_base_ask=n_ask, n_base_act=n_act,
        ask_to_action=ask_to_action, action_to_ask=action_to_ask,
        mean_delta_logitdiff=sum(deltas) / max(1, len(deltas)),
    )
