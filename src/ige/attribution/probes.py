"""Linear probes over stored activations, with the two mandatory canaries.

Rules encoded here (anti-confound design):
- splits are BY PAIR: the two twins of a pair always land on the same side,
  otherwise the probe can key on instruction identity and inflate AUC;
- the permuted-labels canary must sit at chance;
- the positional canary (instruction-final position) must sit at chance,
  since twin activations are bit-identical before the workspace block.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class ProbeResult:
    auc: float
    n_train: int
    n_test: int
    direction: np.ndarray  # unit-norm probe direction in activation space

    def to_dict(self) -> dict:
        return {"auc": self.auc, "n_train": self.n_train, "n_test": self.n_test}


def split_by_pair(pair_ids: list[str], test_frac: float = 0.3, seed: int = 0
                  ) -> tuple[list[int], list[int]]:
    """Index split where twins never straddle train/test."""
    rng = random.Random(seed)
    unique = sorted(set(pair_ids))
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    test_pairs = set(unique[:n_test])
    test_idx = [i for i, p in enumerate(pair_ids) if p in test_pairs]
    train_idx = [i for i, p in enumerate(pair_ids) if p not in test_pairs]
    return train_idx, test_idx


def fit_probe(
    features: torch.Tensor,   # [n, d] activations at one (site, position)
    labels: list[int],        # 1 = ambiguous, 0 = unambiguous
    pair_ids: list[str],
    *,
    seed: int = 0,
    permute_labels: bool = False,
) -> ProbeResult:
    """Train/test logistic probe with pair-respecting split; returns test AUC.

    permute_labels=True runs the canary: labels are shuffled (within the same
    split structure) and the resulting AUC must be ~0.5.
    """
    x = features.float().cpu().numpy()
    y = np.asarray(labels)
    if permute_labels:
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)

    train_idx, test_idx = split_by_pair(pair_ids, seed=seed)
    if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
        raise ValueError("both classes required in each split")

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x[train_idx], y[train_idx])
    scores = clf.decision_function(x[test_idx])
    auc = float(roc_auc_score(y[test_idx], scores))
    norm = np.linalg.norm(clf.coef_[0])
    # degenerate fit (e.g. identical features under a permuted canary): no direction
    direction = clf.coef_[0] / norm if norm > 0 else np.zeros_like(clf.coef_[0])
    return ProbeResult(auc=auc, n_train=len(train_idx), n_test=len(test_idx),
                       direction=direction)


def probe_sweep(
    features_by_site: dict[str, torch.Tensor],
    labels: list[int],
    pair_ids: list[str],
    *,
    seed: int = 0,
) -> dict[str, dict]:
    """Probe + permuted-labels canary per site. The harness refuses to report
    a site whose canary is off-chance (rule: no green canary, no number).
    """
    report: dict[str, dict] = {}
    for site, feats in features_by_site.items():
        res = fit_probe(feats, labels, pair_ids, seed=seed)
        # single-permutation AUC is noisy at small n_test; average 5 permutations
        canary_auc = float(np.mean([
            fit_probe(feats, labels, pair_ids, seed=seed + k, permute_labels=True).auc
            for k in range(5)
        ]))
        ok = abs(canary_auc - 0.5) < 0.15
        report[site] = {
            "auc": res.auc if ok else None,
            "canary_auc": canary_auc,
            "canary_ok": ok,
            **({"suppressed_auc": res.auc} if not ok else {}),
            "n_train": res.n_train,
            "n_test": res.n_test,
        }
    return report
