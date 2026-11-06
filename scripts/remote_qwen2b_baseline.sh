#!/usr/bin/env bash
# Qwen3.5-2B baselines (thinking off, then on) on BIRD dev (full; tuning split = dev minus mini-dev) + Spider dev.
# 1 sample per item (same protocol as the XiYan baselines); generic INSTRUCTION prompt with DDL + 3 sample rows.
#   setsid nohup bash scripts/remote_qwen2b_baseline.sh < /dev/null > q2b_base.log 2>&1 &
set -uo pipefail
cd /workspace/wsp/src
PY=/workspace/wsp/venv/bin/python
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p ../results/p1q_qwen2b
for p in nothink think; do
  out=../results/p1q_qwen2b/qwen35-2b_${p}_bird_dev_spider_dev
  [ -f $out/summary.json ] && { echo "skip $out"; continue; }
  echo "=== $p start $(date -u +%T)"
  $PY generate.py --model qwen35-2b --profile $p --dataset bird_dev,spider_dev --data-root ../data --samples 1 \
      --out $out > $out.gen.log 2>&1 || { echo "GEN_FAILED $p"; tail -20 $out.gen.log; continue; }
  $PY score.py $out --data-root ../data || echo "SCORE_FAILED $p"
  echo "=== $p end $(date -u +%T)"
done
echo Q2B_DONE
