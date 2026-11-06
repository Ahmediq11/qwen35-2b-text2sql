#!/usr/bin/env bash
# Round 2: rejection-sampled self-distillation on BIRD train (discover split), then confirm, then (only if the confirm
# score beats v1's 53.125% vote@8+repair, a rule fixed before seeing round-2 results) the test sets.
set -uo pipefail
cd /workspace/wsp/src
export VLLM_LOGGING_LEVEL=WARNING
PY=../venv/bin/python
until grep -q -E "R2GEN_DONE|Traceback" ../runs/r2_gen.log; do sleep 30; done
grep -q R2GEN_DONE ../runs/r2_gen.log || { echo R2_GEN_FAILED; exit 1; }
$PY r2_data.py ../runs/r2/gen_lora1_bird_discover ../data/sft/r2_bird.jsonl --data-root ../data \
    --model-dir ../models/Qwen3.5-2B --mix ../data/sft/synsql_20k.jsonl --mix-skip 10000 --mix-n 3000 || exit 1
cd ..
venv/bin/python src/train_lora.py --data data/sft/r2_bird.jsonl --out runs/lora_r2 --base models/lora/lora_synsql10k \
    --lr 5e-5 > runs/lora_r2_train.log 2>&1 || { echo R2_TRAIN_FAILED; tail -5 runs/lora_r2_train.log; exit 1; }
grep TRAIN_DONE runs/lora_r2_train.log
cd src
$PY vote_repair.py --path ../models/lora/lora_r2 --ids ../runs/q2b_A/ids_c1_confirm512.json \
    --dataset bird_train,spider_train --data-root ../data --n 8 --out ../runs/vote/c1_lora_r2_8 > ../runs/vote_c1_r2.log 2>&1
pkill -f EngineCore || true
s=$($PY -c "import json;print(json.load(open('../runs/vote/c1_lora_r2_8/summary.json'))['EX']['vote@8+repair'])")
echo "R2 confirm vote@8+repair = $s (v1: 0.53125)"
if $PY -c "import sys; sys.exit(0 if $s > 0.53125 else 1)"; then
  $PY vote_repair.py --path ../models/lora/lora_r2 --ids ../runs/vote/ids_test_birddev_spiderdev.json \
      --dataset bird_dev,spider_dev --data-root ../data --n 8 --out ../results/p4_lora/lora_r2_vote8_repair \
      > ../runs/vote_test_r2.log 2>&1
  pkill -f EngineCore || true
  echo "R2 test done"
else
  echo "R2 not better on confirm: test sets NOT run"
fi
echo R2_ALL_DONE
