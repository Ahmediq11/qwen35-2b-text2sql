#!/usr/bin/env bash
# Wait until a vLLM log shows engine start, an error, or progress; then print the relevant lines.
# usage: bash scripts/wait_log.sh <log> [max_seconds]
log=$1; max=${2:-180}
for ((t = 0; t < max; t += 10)); do
  if [ -s "$log" ] && grep -qE "Traceback|KV cache|Processed prompts" "$log"; then break; fi
  sleep 10
done
grep -vE "it/s\]" "$log" | grep -iE "backend|error|KV cache|concurrency|Traceback" | tail -15
echo ---
tail -c 800 "$log"; echo
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
