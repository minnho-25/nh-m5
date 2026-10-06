"""Stable local launcher for the HuiChain Solara dashboard."""

from __future__ import annotations

import os
import subprocess
import sys


if __name__ == "__main__":
    port = os.environ.get("PORT", "8765")
    command = [
        sys.executable,
        "-m",
        "solara",
        "run",
        "app.py",
        "--production",
        "--host",
        "127.0.0.1",
        "--port",
        port,
    ]
    raise SystemExit(subprocess.call(command))
