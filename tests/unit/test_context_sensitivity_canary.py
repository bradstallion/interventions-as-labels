"""Context-sensitivity canary: identical last-position features across
records must abort the probe run (the OLMo-3 broken-TL-load bug, 2026-07-06:
token replay passed, but activations did not depend on the varying context)."""

import pytest
import torch


def canary(stacked_last: dict[str, torch.Tensor]) -> None:
    # mirror of the blocking check in probe_overask.run
    for s, t in stacked_last.items():
        if float((t - t[0]).abs().max()) == 0.0:
            raise AssertionError(f"context-sensitivity canary failed at {s}")


def test_identical_features_abort():
    row = torch.randn(16)
    with pytest.raises(AssertionError, match="context-sensitivity"):
        canary({"blocks.1.hook_resid_post": row.expand(10, 16).clone()})


def test_varying_features_pass():
    canary({"blocks.1.hook_resid_post": torch.randn(10, 16)})
