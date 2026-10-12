"""Runtime environment helpers: device selection and run metadata.

The device is ALWAYS a config parameter with auto-detection — never
hard-coded. Run metadata (git sha, config
hash) is stamped into every artifact so results are traceable to code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from typing import Any

import torch


def pick_device(requested: str = "auto") -> str:
    """Resolve a device string. 'auto' -> cuda if available, else cpu."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def run_metadata(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Metadata block stamped into every store/artifact."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config_hash": config_hash(config or {}),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
