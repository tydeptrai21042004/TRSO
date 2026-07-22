#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  tests/test_baseline_modules.py \
  tests/test_baseline_integration.py \
  tests/test_corrected_baseline_fidelity.py
