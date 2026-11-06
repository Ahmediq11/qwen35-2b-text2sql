#!/usr/bin/env bash
# After LoRA training: (1) confirm-split 512 paired with c1_orig (same ids/sampling), (2) BIRD dev + Spider dev once.
#   setsid nohup bash scripts/remote_lora_eval.sh <run_name> < /dev/null > runs/lora_eval.log 2>&1 &
set -uo pipefail
cd /workspace/wsp
export VLLM_LOGGING_LEVEL=WARNING
name=$1
until grep -q -E "TRAIN_DONE|Traceback" runs/lora_train.log; do sleep 30; done
grep -q TRAIN_DONE runs/lora_train.log || { echo TRAIN_FAILED; exit 1; }
P=models/lora/$name
cd src
out=../runs/q2b_A/gen/c1_lora_$name
echo "=== confirm start $(date -u +%T)"
../venv/bin/python generate.py --model qwen35-2b --profile think --dataset bird_train,spider_train --data-root ../data \
    --ids ../runs/q2b_A/ids_c1_confirm512.json --samples 1 --out $out --path ../$P --edit lora:$name > $out.gen.log 2>&1 \
    && ../venv/bin/python score.py $out --data-root ../data && ../venv/bin/python ../scripts/paired.py ../runs/q2b_A/gen c1_orig c1_a6 c1_lora_$name
out=../results/p4_lora/${name}_think_bird_dev_spider_dev
mkdir -p ../results/p4_lora
echo "=== test start $(date -u +%T)"
../venv/bin/python generate.py --model qwen35-2b --profile think --dataset bird_dev,spider_dev --data-root ../data \
    --samples 1 --out $out --path ../$P --edit lora:$name > $out.gen.log 2>&1 && ../venv/bin/python score.py $out --data-root ../data
echo "=== end $(date -u +%T)"
echo LORA_EVAL_DONE
