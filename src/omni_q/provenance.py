"""Run provenance — lifted from FALCON-DARPA ``artifacts.py``.

Every run is self-describing: a deterministic ``run_id`` from the config, plus
an environment block (platform, python, package versions, git commit) so a
rerun can be checked against the original.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ["omni-q", "pytest"]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def run_id_for(config: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(config).encode()).hexdigest()[:12]


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def git_commit(root: Path | None = None) -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or Path.cwd()),
            capture_output=True, text=True, timeout=10,
        )
        return res.stdout.strip() or None
    except Exception:
        return None


def build_provenance(generated_by: str, code_root: Path | None = None) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": generated_by,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "package_versions": package_versions(),
        "git_commit": git_commit(code_root),
    }
