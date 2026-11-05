"""Build notebooks/phase0_gsm8k_colab.ipynb: Phase 0 GSM8K check (100 problems, base vs MIT-edited Qwen3.5-0.8B).

    python scripts/build_colab_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Phase 0: GSM8K sanity check, base vs MIT-edited Qwen3.5-0.8B

Reproduces the math check in PLAN.md §3.4: **100 GSM8K test problems** (the first 100 of the test split), thinking on,
the paper's sampling (temp 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, at most 32,768 new tokens, seed 0),
graded on the last `\boxed{}` after `</think>` with math-verify.

**Runtime:** use a GPU runtime (Runtime → Change runtime type). **A100** or **L4** is best: they support bf16, like the
paper's evaluation. A T4 works too, but it runs in fp16 and is much slower.

Rough times for SAMPLES=1 (about 5M generated tokens in total; the base model loops to the token limit almost every
time): A100 about 20–40 min, L4 about 1–1.5 h, T4 several hours.

Results are written to Google Drive (if enabled), so a disconnected session can resume: a model whose `summary_*.json`
already exists is skipped.
""")

code(r"""
# ---- settings ----
N_PROBLEMS = 100        # first N problems of the GSM8K test split
SAMPLES = 1             # samples per problem (the paper used 2 for GSM8K; 2 doubles the time)
MAX_TOKENS = 32768      # the paper's limit
SAVE_TO_DRIVE = True    # write results to /content/drive/MyDrive/sql-wsp/results/p0_gsm8k100

BASE_REPO, BASE_REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
EDIT_REPO, EDIT_REV = "RikiyaT/Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b"

import os, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"],
                     capture_output=True, text=True).stdout)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/sql-wsp/results/p0_gsm8k100"
else:
    OUT = "/content/results/p0_gsm8k100"
os.makedirs(OUT, exist_ok=True)
print("results ->", OUT)
""")

md("## 1. Install (vLLM 0.33.0 / transformers 5.20.2, the versions tested locally). Takes a few minutes.")
code(r"""
!pip install -q "vllm==0.33.0" "transformers==5.20.2" "math-verify[antlr4_13_2]" datasets
import vllm, transformers, torch
print(vllm.__version__, transformers.__version__, torch.__version__, torch.cuda.get_device_name(0),
      "bf16" if torch.cuda.is_bf16_supported() else "fp16 only")
""")

md(r"""
## 2. Download both models at pinned revisions and verify the edit

The expected sha256 values come from `exp8_math/runs/p50c/edit/manifest.json` in `RikiyaT/lm_understand-logs`. The
check also confirms that the edited checkpoint differs from the base in exactly these two tensors. Both repos contain
only safetensors files and configs, with no custom code, so `trust_remote_code` is not needed.
""")
code(r"""
import hashlib, json, struct
from huggingface_hub import snapshot_download

BASE = snapshot_download(BASE_REPO, revision=BASE_REV, local_dir="/content/models/Qwen3.5-0.8B")
EDIT = snapshot_download(EDIT_REPO, revision=EDIT_REV, local_dir="/content/models/Qwen3.5-0.8B-wsp",
                         allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"])
ST = "model.safetensors-00001-of-00001.safetensors"
EXPECTED = {  # name: (base sha256, edited sha256)
    "model.language_model.layers.15.mlp.down_proj.weight":
        ("ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca",
         "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53"),
    "model.language_model.layers.19.mlp.down_proj.weight":
        ("638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e",
         "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2"),
}

def tensor_hashes(path):
    # sha256 of each tensor's raw bytes (bf16 bits), read straight from the safetensors file
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        for name, info in head.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            out[name] = hashlib.sha256(f.read(b - a)).hexdigest()
    return out

hb, he = tensor_hashes(os.path.join(BASE, ST)), tensor_hashes(os.path.join(EDIT, ST))
diff = sorted(k for k in hb if hb[k] != he.get(k))
print(len(hb), "tensors; differing:", diff)
assert set(hb) == set(he), "tensor names differ"
assert diff == sorted(EXPECTED), "unexpected set of edited tensors"
for name, (b, e) in EXPECTED.items():
    assert hb[name] == b and he[name] == e, name
    print("hash OK:", name)
json.dump({"base": {k: hb[k] for k in EXPECTED}, "edited": {k: he[k] for k in EXPECTED}, "differing": diff},
          open(os.path.join(OUT, "hash_check.json"), "w"), indent=1)
""")

md("## 3. GSM8K test problems")
code(r"""
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
PROBLEMS = "/content/gsm8k_test.jsonl"
with open(PROBLEMS, "w") as f:
    for i, r in enumerate(ds.select(range(N_PROBLEMS))):
        f.write(json.dumps({"id": f"gsm8k_test_{i}", "problem": r["question"],
                            "answer": r["answer"].split("####")[-1].strip().replace(",", "")}) + "\n")
print(len(ds), "test problems; using", N_PROBLEMS)
""")

md("## 4. Evaluation script (same protocol as `src/gsm8k_check.py` in the project)")
with open(os.path.join(ROOT, "src", "gsm8k_check.py"), encoding="utf-8") as f:
    code("%%writefile /content/gsm8k_check.py\n" + f.read())

md(r"""
## 5. Run base, then edited

Each model runs in its own process, so GPU memory is freed between runs. A model whose summary already exists in
`OUT` is skipped (resume after a disconnect).
""")
code(r"""
import torch
DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

def run(tag, model):
    if os.path.exists(os.path.join(OUT, f"summary_{tag}.json")):
        print(tag, "already done, skipping")
        return
    cmd = ["python", "/content/gsm8k_check.py", "--model", model, "--tag", tag, "--problems", PROBLEMS,
           "--n-problems", str(N_PROBLEMS), "--samples", str(SAMPLES), "--max-tokens", str(MAX_TOKENS),
           "--dtype", DTYPE, "--mem", "0.90", "--out", OUT]
    with open(os.path.join(OUT, f"log_{tag}.txt"), "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:          # stream the progress bar and the final summary into the cell
            print(line, end="")
            log.write(line)
        assert p.wait() == 0, f"{tag} run failed, see log_{tag}.txt"

run("base", BASE)
""")
code(r"""
run("edited", EDIT)
""")

md("## 6. Results: GATE 0 numbers")
code(r"""
import pandas as pd
S = {t: json.load(open(os.path.join(OUT, f"summary_{t}.json"))) for t in ("base", "edited")}
P = {t: {(r["id"], r["sample"]): r for r in map(json.loads, open(os.path.join(OUT, f"per_item_{t}.jsonl")))}
     for t in ("base", "edited")}
rows = []
for t, s in S.items():
    closed = sum(r["think_closed"] for r in P[t].values()) / len(P[t])
    rows.append({"model": t, "acc": s["acc"], "cut_share": s["cut_share"], "think_closed": closed,
                 "mean_tokens": round(s["mean_tokens"]), "cut_tail_dup8": s["mean_cut_tail_duplicate8"],
                 "gpu_hours": round(s["wall_s"] / 3600, 2), "tok_per_s": round(s["tok_per_s"]), "dtype": s["dtype"]})
table = pd.DataFrame(rows).set_index("model")
display(table)

keys = sorted(set(P["base"]) & set(P["edited"]))
rescues = [k for k in keys if P["edited"][k]["correct"] and not P["base"][k]["correct"]]
losses = [k for k in keys if P["base"][k]["correct"] and not P["edited"][k]["correct"]]
rescued_cut = sum(P["base"][k]["cut"] for k in rescues)
gate = {"n": len(keys), "acc_base": S["base"]["acc"], "acc_edited": S["edited"]["acc"],
        "gain": S["edited"]["acc"] - S["base"]["acc"], "rescues": len(rescues), "losses": len(losses),
        "rescues_where_base_was_cut": rescued_cut,
        "paper_gsm8k": {"unedited": 0.138, "edited_p50c": 0.671},
        "gpu_hours_total": sum(s["wall_s"] for s in S.values()) / 3600, "gpu": torch.cuda.get_device_name(0)}
print(json.dumps(gate, indent=1))
json.dump({"table": rows, "gate0": gate}, open(os.path.join(OUT, "gate0_summary.json"), "w"), indent=1)
""")

md("## 7. Download the results (or copy the Drive folder `sql-wsp/results/p0_gsm8k100` into the project's `results/`)")
code(r"""
import shutil
zip_path = shutil.make_archive("/content/p0_gsm8k100", "zip", OUT)
from google.colab import files
files.download(zip_path)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
for c in nb["cells"]:   # ipynb stores source as a list of lines
    c["source"] = c["source"].splitlines(keepends=True)
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
path = os.path.join(ROOT, "notebooks", "phase0_gsm8k_colab.ipynb")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", path, len(cells), "cells")
