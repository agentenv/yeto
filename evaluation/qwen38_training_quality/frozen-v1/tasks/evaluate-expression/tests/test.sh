#!/usr/bin/env bash
set -eu
mkdir -p /logs/verifier
python3 /tests/verify.py
