"""JumpReLU SAE (Gemma Scope) — minimal in-repo implementation.

No sae_lens dependency: the released params.npz is enough. Keys and math per
the Gemma Scope release (JumpReLU, residual-stream SAEs):

    W_enc [d_model, n_feat]   b_enc [n_feat]
    W_dec [n_feat, d_model]   b_dec [d_model]
    threshold [n_feat]        (per-feature JumpReLU threshold)

    encode:  pre = x @ W_enc + b_enc ;  f = pre * (pre > threshold)
    decode:  x_hat = f @ W_dec + b_dec

Caveat (gate T0, fixed in advance): the SAEs are trained on the PT model,
our activations come from the IT variant — transfer must be validated on our
data (FVU + L0 gates) before any downstream number is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class JumpReluSae:
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    threshold: torch.Tensor

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x.float() @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def fvu(self, x: torch.Tensor) -> float:
        """Fraction of variance unexplained by the reconstruction."""
        x = x.float()
        recon = self.decode(self.encode(x))
        return float(((x - recon) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())

    @classmethod
    def load_gemma_scope(cls, repo: str, layer: int, width: str,
                         device: str = "cpu") -> "JumpReluSae":
        """Load from a gemma-scope HF repo. The '-canonical' repos expose one
        params.npz per (layer, width) at .../canonical/; the base repos expose
        one per L0 target — there we pick the file whose average_l0 is closest
        to the canonical design point (~70)."""
        import numpy as np
        from huggingface_hub import hf_hub_download, list_repo_files

        prefix = f"layer_{layer}/width_{width}/"
        if repo.endswith("-canonical"):
            path = prefix + "canonical/params.npz"
        else:
            files = [f for f in list_repo_files(repo)
                     if f.startswith(prefix) and f.endswith("params.npz")]
            if not files:
                raise FileNotFoundError(f"no params.npz under {prefix} in {repo}")
            path = min(files, key=lambda f: abs(int(f.split("average_l0_")[1].split("/")[0]) - 70))
        local = hf_hub_download(repo, path)
        params = np.load(local)
        t = {k: torch.tensor(params[k], dtype=torch.float32, device=device) for k in params}
        return cls(W_enc=t["W_enc"], b_enc=t["b_enc"], W_dec=t["W_dec"],
                   b_dec=t["b_dec"], threshold=t["threshold"])


def topk_by_separation(feats: torch.Tensor, labels: list[int],
                       idx: list[int], k: int) -> list[int]:
    """Top-k features by standardized class-mean separation, computed on the
    given (train) indices only — selection never sees test episodes."""
    f = feats[idx].float()
    y = torch.tensor([labels[i] for i in idx])
    mu1, mu0 = f[y == 1].mean(0), f[y == 0].mean(0)
    sd = f.std(0) + 1e-6
    return torch.argsort((mu1 - mu0).abs() / sd, descending=True)[:k].tolist()


def decoder_alignment(direction: torch.Tensor, sae: JumpReluSae) -> torch.Tensor:
    """|cos| between a (dense, unit) direction and each decoder row — how much
    each feature writes along that axis."""
    dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
    return (dec @ direction.float()).abs()


def clamp_features(x: torch.Tensor, sae: JumpReluSae, feat_idx: list[int],
                   targets: torch.Tensor) -> torch.Tensor:
    """Move x so that the chosen features read `targets` instead of their
    current values, via the decoder directions (standard SAE steering):
    x' = x + sum_f (t_f - a_f) * W_dec[f]. Everything else untouched."""
    a = sae.encode(x)[..., feat_idx]
    delta = (targets - a) @ sae.W_dec[feat_idx]
    return x + delta
