#!/usr/bin/env bash
# waits for the GPU to be free (random-control pass), then trains
cd /workspace/wsp
until grep -q TRACKA_GEN_DONE runs/q2b_A/g1r.log 2>/dev/null; do sleep 20; done
venv/bin/python src/train_lora.py --data data/sft/synsql_10k.jsonl --out runs/lora_synsql10k
