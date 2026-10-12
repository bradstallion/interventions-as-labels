import pytest
import torch

from ige.utils.env import config_hash, pick_device, run_metadata
from ige.utils.seed import set_seed

pytestmark = pytest.mark.unit


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(8)
    set_seed(123)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = torch.randn(8)
    set_seed(2)
    b = torch.randn(8)
    assert not torch.equal(a, b)


def test_pick_device_explicit_passthrough():
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda:1") == "cuda:1"


def test_pick_device_auto_resolves():
    assert pick_device("auto") in ("cuda", "cpu")


def test_config_hash_stable_and_order_insensitive():
    h1 = config_hash({"a": 1, "b": [1, 2]})
    h2 = config_hash({"b": [1, 2], "a": 1})
    assert h1 == h2
    assert h1 != config_hash({"a": 2, "b": [1, 2]})


def test_run_metadata_has_required_keys():
    md = run_metadata({"x": 1})
    for key in ("created_utc", "git_sha", "config_hash", "torch_version"):
        assert key in md
