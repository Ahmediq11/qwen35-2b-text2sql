"""Round-2 SFT data by rejection sampling (self-distillation on in-domain data).

From a scored generation run with several samples per question (the LoRA v1 model on BIRD train, discover split),
keep, for every question, the SHORTEST sample whose SQL is execution-correct against the gold query, and use the
model's own text (reasoning inside <think> + the ```sql answer) as the target. Questions with no correct sample are
skipped (we never train on gold SQL without reasoning). Optionally mix in unseen SynSQL rows to keep breadth.

    python src/r2_data.py runs/r2/gen_lora1_bird_discover data/sft/r2_bird.jsonl \
        --mix data/sft/synsql_20k.jsonl --mix-skip 10000 --mix-n 3000
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import prompts as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model-dir", default="models/Qwen3.5-2B")
    ap.add_argument("--mix")
    ap.add_argument("--mix-skip", type=int, default=0, help="skip the first rows of --mix (already used in v1)")
    ap.add_argument("--mix-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    items = {it["id"]: it for n in cfg["dataset"].split(",") for it in D.load(a.data_root, n)}
    by_id = defaultdict(list)
    for l in open(os.path.join(a.run, "per_item.jsonl")):
        r = json.loads(l)
        by_id[r["id"]].append(r)
    rows, stats = [], {"questions": len(by_id), "with_correct": 0, "samples": 0, "correct_samples": 0}
    for pid, rs in by_id.items():
        stats["samples"] += len(rs)
        ok = [r for r in rs if r["outcome"] == "correct" and "</think>" in r["text"]]
        stats["correct_samples"] += len(ok)
        if not ok:
            continue
        stats["with_correct"] += 1
        best = min(ok, key=lambda r: r["tokens"])
        it = items[pid]
        prompt = tok.apply_chat_template(P.messages(it, "qwen"), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        rows.append({"prompt": prompt, "target": best["text"].strip(), "db_id": it["db_id"], "source": "r2_self",
                     "id": pid, "tokens": best["tokens"]})
    if a.mix and a.mix_n:
        mix = [json.loads(l) for i, l in enumerate(open(a.mix)) if i >= a.mix_skip][:a.mix_n]
        for m in mix:
            m["source"] = "synsql"
        rows += mix
        stats["mixed_synsql"] = len(mix)
    random.Random(a.seed).shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    self_rows = [r for r in rows if r["source"] == "r2_self"]
    stats.update(kept=len(rows), self_examples=len(self_rows),
                 mean_self_tokens=sum(r["tokens"] for r in self_rows) / max(1, len(self_rows)),
                 pass_at_k=stats["with_correct"] / max(1, stats["questions"]))
    json.dump(stats, open(a.out.replace(".jsonl", ".stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
