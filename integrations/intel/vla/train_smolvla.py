"""lerobot-train with Windows-safe checkpoint links (os.symlink needs a
privilege here; the 'last' link becomes a text pointer instead)."""
import os
import sys
from pathlib import Path

_real_symlink = os.symlink


def _symlink(src, dst, *a, **k):
    try:
        _real_symlink(src, dst, *a, **k)
    except OSError:
        Path(str(dst) + ".txt").write_text(str(src))


os.symlink = _symlink
from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
