"""Seeding and determinism helpers.

Every experiment entrypoint MUST call set_seed() before touching the model.
Reproducibility is a gate requirement.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_algorithms: bool = False) -> int:
    """Seed python, numpy and torch (all devices). Returns the seed.

    deterministic_algorithms=True additionally forces torch deterministic
    kernels (slower; some ops unsupported on CUDA — use for debugging
    non-reproducible results, not for every run).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        # cuBLAS needs this env var to make matmuls deterministic.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    return seed
