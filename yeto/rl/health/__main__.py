"""``python -m yeto.rl.health RUN_DIR``: the backend-independent health entry."""

import sys

from yeto.rl.health.entry import main

sys.exit(main())
