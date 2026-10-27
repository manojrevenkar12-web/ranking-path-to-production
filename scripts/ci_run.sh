#!/usr/bin/env bash
# Runs a CI step's command. If it fails, the end of its output becomes one error annotation, which shows on the
# run's summary page without opening the job log.
#
#     scripts/ci_run.sh pytest -q -m vespa
set -uo pipefail
out=$(mktemp)
"$@" 2>&1 | tee "$out"
code=${PIPESTATUS[0]}
if [ "$code" -ne 0 ]; then
  body=$(tail -n "${CI_RUN_TAIL:-60}" "$out" | sed -e 's/%/%25/g' -e 's/\r/%0D/g' | awk 'BEGIN { ORS = "%0A" } { print }')
  echo "::error title=${1} exited with ${code}::${body}"
fi
exit "$code"
