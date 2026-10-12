"""Probe tests on synthetic activations with a PLANTED direction: the probe
MUST find a signal that is there by construction, and must NOT find one in
noise or under permuted labels."""

import numpy as np
import pytest
import torch

from ige.attribution.probes import fit_probe, probe_sweep, split_by_pair

pytestmark = pytest.mark.unit

D = 32
N_PAIRS = 40


def _planted(strength: float, seed: int = 0):
    """Synthetic twins: shared pair noise, label-dependent shift on a planted
    unit direction. Returns (features, labels, pair_ids, direction)."""
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(D, generator=g)
    direction /= direction.norm()
    feats, labels, pair_ids = [], [], []
    for p in range(N_PAIRS):
        base = torch.randn(D, generator=g)
        for label in (1, 0):
            noise = 0.3 * torch.randn(D, generator=g)
            shift = strength * direction if label else torch.zeros(D)
            feats.append(base + noise + shift)
            labels.append(label)
            pair_ids.append(f"pair-{p:03d}")
    return torch.stack(feats), labels, pair_ids, direction


def test_probe_finds_planted_direction():
    # strength 5 vs per-dim base noise sigma=1 -> d' ~ 5, theoretical AUC > 0.999
    feats, labels, pair_ids, direction = _planted(strength=5.0)
    res = fit_probe(feats, labels, pair_ids)
    assert res.auc > 0.95
    cos = abs(float(np.dot(res.direction, direction.numpy())))
    assert cos > 0.7  # recovered direction aligned with the planted one


def test_probe_at_chance_on_pure_noise():
    feats, labels, pair_ids, _ = _planted(strength=0.0)
    res = fit_probe(feats, labels, pair_ids)
    assert 0.3 < res.auc < 0.7


def test_permuted_labels_canary_at_chance():
    feats, labels, pair_ids, _ = _planted(strength=5.0)
    canary = fit_probe(feats, labels, pair_ids, permute_labels=True)
    assert 0.3 < canary.auc < 0.7


def test_split_by_pair_never_splits_twins():
    pair_ids = [f"p{i//2}" for i in range(20)]
    train, test = split_by_pair(pair_ids, seed=3)
    train_pairs = {pair_ids[i] for i in train}
    test_pairs = {pair_ids[i] for i in test}
    assert not (train_pairs & test_pairs)
    assert len(train) + len(test) == 20


def test_probe_sweep_suppresses_bad_canary_sites():
    feats, labels, pair_ids, _ = _planted(strength=5.0)
    report = probe_sweep({"site_a": feats}, labels, pair_ids)
    assert report["site_a"]["canary_ok"] and report["site_a"]["auc"] > 0.9


def test_overask_content_features_and_labels():
    from ige.experiments.probe_overask import behavior_label, content_features

    def rec(family, amb, outcome, clarify_dim="size", p=1):
        items = [
            {"id": "b1", "attrs": {"color": "red", "size": "small"}},
            {"id": "b2", "attrs": {"color": "blue", "size": "large"}},
            {"id": "b3", "attrs": {"color": "green", "size": "tiny"}},
        ]
        return {"scenario": {"family": family, "ambiguous": amb, "clarify_dim": clarify_dim,
                             "paraphrase_idx": p, "instruction": "Bring me the red book.",
                             "items": items, "target_id": "b1"},
                "outcome": outcome}

    assert behavior_label(rec("fetch", False, "overask")) == 1
    assert behavior_label(rec("fetch", False, "correct_act")) == 0
    assert behavior_label(rec("fetch", True, "correct_clarify")) is None
    assert behavior_label(rec("fetch", False, "invalid")) is None

    feats = content_features([rec("fetch", False, "overask"), rec("files", False, "correct_act")])
    assert feats.shape[0] == 2 and feats.shape[1] >= 7
    assert not torch.equal(feats[0], feats[1])


def test_project_to_moves_only_the_direction_coordinate():
    from ige.attribution.direction_patch import project_to

    g = torch.Generator().manual_seed(0)
    h = torch.randn(16, generator=g)
    d = torch.randn(16, generator=g); d /= d.norm()
    h2 = project_to(h, d, target=3.0)
    assert torch.allclose(h2 @ d, torch.tensor(3.0), atol=1e-5)
    # orthogonal complement untouched
    ortho = h - (h @ d) * d
    ortho2 = h2 - (h2 @ d) * d
    assert torch.allclose(ortho, ortho2, atol=1e-5)


def test_direction_patch_result_rates_and_style_variant():
    from ige.attribution.direction_patch import DirectionPatchResult
    from ige.experiments.m4prep_overask import ASK_SUFFIX, style_variant

    r = DirectionPatchResult(site="s", alpha_label="a", n=10, n_base_ask=8,
                             n_base_act=2, ask_to_action=4, action_to_ask=1,
                             mean_delta_logitdiff=-1.0)
    assert r.flip_rate == 0.5 and r.induce_rate == 0.5
    assert r.to_dict()["flip_rate"] == 0.5

    msgs = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "TASK"}]
    v = style_variant(msgs, ASK_SUFFIX)
    assert v[0]["content"].startswith("RULES") and "ALWAYS" in v[0]["content"]
    assert v[1] == msgs[1]           # user turn untouched
    assert msgs[0]["content"] == "RULES"  # input not mutated
