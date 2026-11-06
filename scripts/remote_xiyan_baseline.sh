#!/usr/bin/env bash
# XiYan baselines with the model card's protocol (1 sample, temperature 0.1, DDL schema) on BIRD dev (full, the
# published set) and Spider dev. Reproduction check against the card's DDL numbers (BIRD dev: 7B 57.43, 3B 52.09).
#   setsid nohup bash scripts/remote_xiyan_baseline.sh < /dev/null > xiyan_base.log 2>&1 &
set -uo pipefail
cd /workspace/wsp/src
PY=/workspace/wsp/venv/bin/python
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p ../results/p1x_xiyan
for m in xiyan-7b xiyan-3b; do
  out=../results/p1x_xiyan/${m}_bird_dev_spider_dev
  [ -f $out/summary.json ] && { echo "skip $out"; continue; }
  echo "=== $m start $(date -u +%T)"
  $PY generate.py --model $m --profile xiyan --dataset bird_dev,spider_dev --data-root ../data --samples 1 \
      --out $out > $out.gen.log 2>&1 || { echo "GEN_FAILED $m"; tail -20 $out.gen.log; continue; }
  $PY score.py $out --data-root ../data || echo "SCORE_FAILED $m"
  echo "=== $m end $(date -u +%T)"
done
echo XIYAN_DONE
