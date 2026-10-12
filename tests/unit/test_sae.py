"""JumpReLU SAE math + selection/steering helpers (m6_sae predictions):
- encode applies the per-feature JumpReLU threshold (not a plain ReLU);
- decode/fvu algebra; perfect dictionary -> FVU ~ 0;
- clamp_features moves ONLY along the chosen decoder rows and lands the
  chosen features on their targets when the dictionary is orthonormal;
- topk_by_separation finds a planted feature using train indices only."""

import torch

from ige.attribution.sae import (JumpReluSae, clamp_features,
                                 decoder_alignment, topk_by_separation)


def make_orthonormal_sae(d: int = 8, f: int = 8, thr: float = 0.5) -> JumpReluSae:
    eye = torch.eye(d)
    return JumpReluSae(W_enc=eye.clone(), b_enc=torch.zeros(f),
                       W_dec=eye.clone(), b_dec=torch.zeros(d),
                       threshold=torch.full((f,), thr))


def test_jumprelu_threshold_gates_small_activations():
    sae = make_orthonormal_sae(thr=0.5)
    x = torch.tensor([[0.4, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    f = sae.encode(x)
    assert f[0, 0] == 0.0          # below threshold: killed, not just shrunk
    assert f[0, 1] == 0.6          # above threshold: passes UNCHANGED (jump, not shift)
    assert f[0, 2] == 0.0          # negative never passes


def test_perfect_dictionary_reconstructs():
    sae = make_orthonormal_sae(thr=0.0)
    x = torch.rand(20, 8) + 0.1    # all positive -> all features active
    assert sae.fvu(x) < 1e-10


def test_clamp_features_hits_targets_and_touches_nothing_else():
    sae = make_orthonormal_sae(thr=0.0)
    x = torch.rand(5, 8) + 0.1
    fs = [1, 3]
    targets = torch.tensor([2.0, 0.0])
    x2 = clamp_features(x, sae, fs, targets)
    f2 = sae.encode(x2)
    assert torch.allclose(f2[:, 1], torch.full((5,), 2.0))
    assert torch.allclose(f2[:, 3], torch.zeros(5))
    untouched = [i for i in range(8) if i not in fs]
    assert torch.allclose(x2[:, untouched], x[:, untouched])


def test_topk_finds_planted_feature():
    torch.manual_seed(0)
    feats = torch.rand(100, 32)
    labels = [i % 2 for i in range(100)]
    feats[:, 7] += 3.0 * torch.tensor(labels, dtype=torch.float32)
    sel = topk_by_separation(feats, labels, list(range(60)), k=3)
    assert sel[0] == 7


def test_decoder_alignment_prefers_matching_row():
    sae = make_orthonormal_sae()
    d = torch.zeros(8)
    d[2] = 1.0
    a = decoder_alignment(d, sae)
    assert int(torch.argmax(a)) == 2
