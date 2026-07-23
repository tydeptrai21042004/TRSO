#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python -m tools.audit_universal_release --output test_reports/universal_release_audit.json
