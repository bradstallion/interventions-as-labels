"""annotate_irr: blind sampling is balanced and deterministic; the summary
computes per-policy burden, IRR and kappa from the annotation file format."""

import pytest

from ige.overseer.annotate_irr import build_sample, episode_key, summarize


def make_records(n_per=10):
    """Mirror of the episodes.jsonl format written by steer_eval.run():
    'none' episodes all overask (rule intervenes), 'steer_gated' all clean."""
    recs = []
    for pol, intervene in (("none", True), ("steer_gated", False)):
        for fam in ("email", "file"):
            for i in range(n_per):
                recs.append({
                    "policy": pol,
                    "scenario": {"scenario_id": f"{fam}-{i:04d}", "family": fam},
                    "messages_per_step": [[{"role": "user", "content": "task"}]],
                    "raw_replies": ["ASK: which?" if intervene else "ACTION: d1"],
                    "outcome": "overask" if intervene else "correct_act",
                    "verdict": {"intervene": intervene,
                                "type": "unnecessary-clarification" if intervene else ""},
                })
    return recs


POLICIES = ["none", "steer_gated"]


def test_build_sample_balanced_and_deterministic():
    recs = make_records()
    sample = build_sample(recs, POLICIES, n=20, seed=0)
    assert len(sample) == 20
    per_pol = {p: sum(r["policy"] == p for r in sample) for p in POLICIES}
    assert per_pol == {"none": 10, "steer_gated": 10}
    per_fam = {f: sum(r["scenario"]["family"] == f for r in sample)
               for f in ("email", "file")}
    assert per_fam == {"email": 10, "file": 10}
    # deterministic (resume relies on it), and actually shuffled across policies
    again = build_sample(recs, POLICIES, n=20, seed=0)
    assert [episode_key(r) for r in again] == [episode_key(r) for r in sample]
    assert [r["policy"] for r in sample] != ["none"] * 10 + ["steer_gated"] * 10


def test_build_sample_missing_policy_raises():
    with pytest.raises(ValueError, match="no episodes"):
        build_sample(make_records(), ["none", "nope"], n=20, seed=0)


def test_summarize_full_agreement():
    recs = make_records()
    sample = build_sample(recs, POLICIES, n=20, seed=0)
    # human agrees with rule-v2 everywhere
    ann = {episode_key(r): {"episode_key": episode_key(r),
                            "intervene": r["verdict"]["intervene"]}
           for r in sample}
    s = summarize(recs, sample, ann, POLICIES)
    assert s["complete"]
    assert s["kappa_human_rule"] == pytest.approx(1.0)
    none, gated = s["policies"]["none"], s["policies"]["steer_gated"]
    assert none["human_intervene"] == 10 and gated["human_intervene"] == 0
    assert gated["irr_human"] == pytest.approx(gated["irr_rule"])
    assert gated["irr_human"] > 0  # steering removed every intervention
    assert gated["human_burden"] < none["human_burden"]


def test_summarize_partial_is_flagged():
    recs = make_records()
    sample = build_sample(recs, POLICIES, n=20, seed=0)
    ann = {episode_key(r): {"episode_key": episode_key(r), "intervene": False}
           for r in sample[:5]}
    s = summarize(recs, sample, ann, POLICIES)
    assert not s["complete"]
    assert "kappa_human_rule" not in s  # <10 annotated: no kappa printed


def test_summarize_human_disagrees():
    recs = make_records()
    sample = build_sample(recs, POLICIES, n=20, seed=0)
    # human never intervenes: kappa 0 side, IRR_human 0 (no burden reduction)
    ann = {episode_key(r): {"episode_key": episode_key(r), "intervene": False}
           for r in sample}
    s = summarize(recs, sample, ann, POLICIES)
    assert s["kappa_human_rule"] == pytest.approx(0.0)
    assert s["policies"]["steer_gated"]["irr_rule"] > 0
    # human burden identical across policies -> human IRR ~ 0 (watch-only cost)
    assert abs(s["policies"]["steer_gated"]["irr_human"]) < 0.35
