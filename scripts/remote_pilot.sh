#!/usr/bin/env bash
# Phase 1 pilot on PRACTICE data (discover split; no test items): every baseline configuration on a few items, to
# validate the code path and measure tokens per answer and throughput before the full matrix.
#   setsid nohup bash scripts/remote_pilot.sh < /dev/null > pilot.log 2>&1 &
set -uo pipefail
cd /workspace/wsp/src
PY=/workspace/wsp/venv/bin/python
export VLLM_LOGGING_LEVEL=WARNING
$PY - <<'EOF'
import json, hashlib
s = json.load(open("../data/practice_splits.json"))
disc = sorted(s["discover"], key=lambda i: hashlib.sha256(i.encode()).hexdigest())
pick = lambda src, n: [i for i in disc if i.startswith(src)][:n]
json.dump(pick("spider_train", 25) + pick("bird_train", 25), open("../data/pilot50_ids.json", "w"))
json.dump(pick("spider_train", 10) + pick("bird_train", 10), open("../data/pilot20_ids.json", "w"))
EOF
run() {  # model profile ids
  out=../results/p1_pilot/$1_$2
  [ -f $out/summary.json ] && { echo "skip $out"; return; }
  echo "=== $1 $2 start $(date -u +%T)"
  $PY generate.py --model $1 --profile $2 --dataset spider_train,bird_train --data-root ../data --ids $3 \
      --samples 2 --out $out > $out.gen.log 2>&1 || { echo "GEN_FAILED $1 $2"; tail -20 $out.gen.log; return; }
  $PY score.py $out --data-root ../data || echo "SCORE_FAILED $1 $2"
  echo "=== $1 $2 end $(date -u +%T)"
}
mkdir -p ../results/p1_pilot
for m in qwen35-0.8b qwen35-0.8b-wsp qwen35-2b; do run $m nothink ../data/pilot50_ids.json; done
run xiyan-3b xiyan ../data/pilot50_ids.json
for m in qwen35-0.8b qwen35-0.8b-wsp qwen35-2b; do run $m think ../data/pilot20_ids.json; done
echo PILOT_DONE
