#!/usr/bin/env bash
set -euo pipefail
python -m tools.run_hparam_sweep "$@"
