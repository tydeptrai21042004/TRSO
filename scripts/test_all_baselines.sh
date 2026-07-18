#!/usr/bin/env bash
set -euo pipefail

python -m unittest tests.test_baseline_modules
python -m unittest tests.test_baseline_integration
