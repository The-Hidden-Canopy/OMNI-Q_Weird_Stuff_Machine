#!/usr/bin/env bash
# Omni Q mock demo (OQ-001): Observe -> Plan -> Manipulate -> Verify -> Receipt.
#
# Runs the three observable states from DEMO.md with fake providers only:
#   1. NORMAL   2. CONSTRAINT CHANGE   3. FAILURE / WORLD CHANGE
#
# No hardware, no third-party deps. Real providers land per the BACKLOG.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python py; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
fi
[ -n "$PY" ] || { echo "no python interpreter found; set \$PYTHON" >&2; exit 1; }

PYTHONPATH=src exec "$PY" -m omni_q.demo
