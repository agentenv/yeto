#!/usr/bin/env python3
"""Thin entry for the RL elastic resource benchmark suite.

See docs/RL_ELASTIC_BENCHMARK.md. Logic lives in yeto/rl/elastic_benchmark/.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yeto.rl.elastic_benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
