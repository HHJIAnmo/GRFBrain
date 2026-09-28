#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Harness Initialization ==="

echo "=== python -m pytest -q ==="
python -m pytest -q

echo "=== python -m eeg_cgfm.cli.smoke ==="
python -m eeg_cgfm.cli.smoke

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read feature_list.json to see current feature state"
echo "2. Pick ONE unfinished feature to work on"
echo "3. Implement only that feature"
echo "4. Re-run verification before claiming done"
