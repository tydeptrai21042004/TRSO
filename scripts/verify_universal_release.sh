#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
python -m tools.preflight_paper_baselines --json test_reports/paper_baselines_preflight.json
python -m tools.preflight_trso --json test_reports/trso_preflight_universal.json
python -m tools.preflight_trso_v2 --json test_reports/trso_v2_preflight.json
python -m tools.preflight_trso_v3 --json test_reports/trso_v3_preflight.json
python -m tools.audit_universal_release --output test_reports/universal_release_audit.json
