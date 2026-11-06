#!/usr/bin/env bash
# Phase 0 on a rented GPU box: download both models at pinned revisions, verify the p50c edit, then run the
# 100-problem GSM8K check for base and edited. Resumable: a model whose summary exists is skipped.
#   nohup bash scripts/remote_phase0.sh > run.log 2>&1 &
set -euo pipefail
cd /workspace/wsp
PY=/workspace/wsp/venv/bin/python
OUT=results/p0_gsm8k100
N=${N_PROBLEMS:-100}
SAMPLES=${SAMPLES:-1}
mkdir -p models "$OUT"

until grep -q '^EXIT=' install.log 2>/dev/null; do echo "waiting for install..."; sleep 20; done
grep -q '^EXIT=0' install.log || { echo "install failed"; tail -30 install.log; exit 1; }
$PY -c 'import vllm, torch, transformers; print("vllm", vllm.__version__, "torch", torch.__version__, "transformers", transformers.__version__, torch.cuda.get_device_name(0))'

export HF_HUB_ENABLE_HF_TRANSFER=1
$PY - <<'EOF'
from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17", local_dir="models/Qwen3.5-0.8B")
s("RikiyaT/Qwen3.5-0.8B-weight-space-programming", revision="d1671a178705bcab7644806d23c535517f35723b",
  local_dir="models/Qwen3.5-0.8B-wsp", allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"])
EOF

ST=model.safetensors-00001-of-00001.safetensors
$PY src/edits.py verify models/Qwen3.5-0.8B/$ST edit_p50c | tee "$OUT/hash_check.json"
$PY - <<EOF
import sys; sys.path.insert(0, "src")
import edits as E
a, b = E.load("models/Qwen3.5-0.8B/$ST"), E.load("models/Qwen3.5-0.8B-wsp/$ST")
diff = sorted(n for n in a if E.sha256(a[n][1]) != E.sha256(b[n][1]))
print(len(a), "tensors; HF edited differs from base in:", diff)
assert set(a) == set(b) and diff == ["model.language_model.layers.15.mlp.down_proj.weight",
                                     "model.language_model.layers.19.mlp.down_proj.weight"]
_, rows = E.apply_edit("models/Qwen3.5-0.8B/$ST", "edit_p50c")
for r in rows:
    assert r["edited_sha256"] == E.sha256(b[r["name"]][1]), r["name"]
print("HF edited tensors == rebuilt p50c edit: OK")
EOF

export VLLM_LOGGING_LEVEL=WARNING
for tag in base edited; do
  model=models/Qwen3.5-0.8B; [ "$tag" = edited ] && model=models/Qwen3.5-0.8B-wsp
  if [ -f "$OUT/summary_$tag.json" ]; then echo "$tag done, skipping"; continue; fi
  echo "=== $tag start $(date -u +%FT%TZ)"
  $PY src/gsm8k_check.py --model "$model" --tag "$tag" --problems data/gsm8k/test100.jsonl --n-problems "$N" \
      --samples "$SAMPLES" --max-tokens 32768 --dtype bfloat16 --mem 0.90 --out "$OUT" > "$OUT/log_$tag.txt" 2>&1
  echo "=== $tag end $(date -u +%FT%TZ)"
done
echo ALL_DONE
