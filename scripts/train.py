"""CLI entry point for sentiment model training."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sentiment_analysis.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
