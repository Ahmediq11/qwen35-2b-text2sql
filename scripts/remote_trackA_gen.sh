#!/usr/bin/env bash
# Track A paired generation passes on a fresh discover subset. Args: ids file, then "tag:state" pairs.
#   setsid nohup bash scripts/remote_trackA_gen.sh runs/q2b_A/ids.json orig:original a6:edits/x ... < /dev/null > log 2>&1 &
set -uo pipefail
cd /workspace/wsp
export VLLM_LOGGING_LEVEL=WARNING
ids=$1; shift
for ts in "$@"; do
  tag=${ts%%:*}; st=${ts#*:}
  echo "=== $tag ($st) start $(date -u +%T)"
  venv/bin/python src/fwd_tool.py --run runs/q2b_A generate --state $st --ids $ids --tag $tag 2>&1 | grep '"type"' | head -1
done
echo TRACKA_GEN_DONE
