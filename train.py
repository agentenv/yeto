#!/usr/bin/env python3
"""Thin wrapper so `python3 train.py ...` keeps working from a checkout;
the implementation lives in yeto.cli (installed as the `yeto` console
script)."""

import sys

from yeto.cli import main

if __name__ == "__main__":
    sys.exit(main())
