"""load_hooked routing: bridge flag selects TransformerBridge.boot_transformers,
default selects HookedTransformer.from_pretrained; hf_model is forwarded on
both paths (edit_eval loads merged adapters through here)."""

import sys
import types

from ige.hooks.load import load_hooked


def _stub_tl(monkeypatch, calls):
    tl = types.ModuleType("transformer_lens")

    class HT:
        @staticmethod
        def from_pretrained(name, **kw):
            calls.append(("legacy", name, kw))
            return "legacy-model"

    tl.HookedTransformer = HT

    mb = types.ModuleType("transformer_lens.model_bridge")

    class BridgeModel:
        kind = "bridge-model"

        def __init__(self, device_type="cpu"):
            self._dev = device_type

        def parameters(self):
            dev = self._dev

            class P:
                class device:
                    type = None
            P.device.type = dev
            yield P()

    class TB:
        @staticmethod
        def boot_transformers(name, **kw):
            calls.append(("bridge", name, kw))
            hf = kw.get("hf_model")
            dev = getattr(hf, "moved_to", None) or "cpu"
            return BridgeModel(device_type=dev)

    mb.TransformerBridge = TB
    tl.model_bridge = mb
    monkeypatch.setitem(sys.modules, "transformer_lens", tl)
    monkeypatch.setitem(sys.modules, "transformer_lens.model_bridge", mb)


def test_default_uses_legacy_port(monkeypatch):
    calls = []
    _stub_tl(monkeypatch, calls)
    out = load_hooked("m", "cpu", None)
    assert out == "legacy-model"
    kind, name, kw = calls[0]
    assert kind == "legacy" and "hf_model" not in kw


def test_bridge_flag_uses_bridge(monkeypatch):
    calls = []
    _stub_tl(monkeypatch, calls)
    out = load_hooked("m", "cpu", None, bridge=True)
    assert getattr(out, "kind", None) == "bridge-model"
    assert calls[0][0] == "bridge"


class FakeHF:
    """Minimal hf_model stub: supports .to() like a real module."""

    def __init__(self):
        self.moved_to = None

    def to(self, device=None, dtype=None):
        self.moved_to = device
        return self


def test_hf_model_forwarded_both_paths(monkeypatch):
    calls = []
    _stub_tl(monkeypatch, calls)
    hf1, hf2 = FakeHF(), FakeHF()
    load_hooked("m", "cpu", None, hf_model=hf1)
    load_hooked("m", "cpu", None, bridge=True, hf_model=hf2)
    assert calls[0][2]["hf_model"] is hf1
    assert calls[1][2]["hf_model"] is hf2


def test_no_processing_routes_to_raw_loader(monkeypatch):
    calls = []
    _stub_tl(monkeypatch, calls)
    sys.modules["transformer_lens"].HookedTransformer.from_pretrained_no_processing = (
        lambda name, **kw: calls.append(("raw", name, kw)) or "raw-model")
    out = load_hooked("m", "cpu", None, no_processing=True)
    assert out == "raw-model" and calls[0][0] == "raw"


def test_bridge_moves_provided_hf_model_to_device(monkeypatch):
    calls = []
    _stub_tl(monkeypatch, calls)
    hf = FakeHF()
    load_hooked("m", "cuda", None, bridge=True, hf_model=hf)
    assert hf.moved_to == "cuda"


def test_bridge_cpu_landing_on_cuda_request_raises(monkeypatch):
    import pytest

    calls = []
    _stub_tl(monkeypatch, calls)

    class StubbornHF(FakeHF):
        def to(self, device=None, dtype=None):
            return self  # never actually moves

    with pytest.raises(AssertionError, match="landed on cpu"):
        load_hooked("m", "cuda", None, bridge=True, hf_model=StubbornHF())
