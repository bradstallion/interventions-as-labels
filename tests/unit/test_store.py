import json

import pytest
import torch

from ige.hooks.store import ActivationStore

pytestmark = pytest.mark.unit


def _fake_activations():
    g = torch.Generator().manual_seed(0)
    return {
        "blocks.0.hook_resid_post": {
            0: torch.randn(5, 16, generator=g),
            1: torch.randn(9, 16, generator=g),  # different seq len: no padding stored
        },
        "blocks.1.hook_resid_post": {
            0: torch.randn(5, 16, generator=g),
            1: torch.randn(9, 16, generator=g),
        },
    }


def test_roundtrip_exact(tmp_path):
    acts = _fake_activations()
    store = ActivationStore.write(tmp_path / "s", acts, meta={"model": "fake"})
    for site, per_prompt in acts.items():
        for idx, t in per_prompt.items():
            assert torch.equal(store.get(idx, site), t)


def test_reopen_from_disk(tmp_path):
    acts = _fake_activations()
    ActivationStore.write(tmp_path / "s", acts, meta={"model": "fake", "seed": 7})
    store = ActivationStore(tmp_path / "s")
    assert store.meta["seed"] == 7
    assert store.sites == sorted(acts.keys())
    assert store.prompt_indices("blocks.0.hook_resid_post") == [0, 1]
    assert store.shape(1, "blocks.0.hook_resid_post") == [9, 16]
    assert torch.equal(store.get(1, "blocks.1.hook_resid_post"), acts["blocks.1.hook_resid_post"][1])


def test_write_refuses_overwrite(tmp_path):
    acts = _fake_activations()
    ActivationStore.write(tmp_path / "s", acts, meta={})
    with pytest.raises(FileExistsError):
        ActivationStore.write(tmp_path / "s", acts, meta={})


def test_index_is_valid_json(tmp_path):
    ActivationStore.write(tmp_path / "s", _fake_activations(), meta={})
    index = json.loads((tmp_path / "s" / "index.json").read_text())
    assert index["blocks.0.hook_resid_post"]["1"] == [9, 16]
