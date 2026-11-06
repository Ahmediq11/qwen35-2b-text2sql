"""Matched productive/repetitive contrast texts from a scored generation run (CPU only, no model calls).

For every cut (looping) answer we locate the loop onset with a word 8-gram duplicate test and make a matched pair from
the SAME answer: a productive span just before the onset and a repetitive span inside the loop (same problem, same
prompt, identical text up to the earlier cut point). Finished-correct answers add a second productive set (a span from
the middle of their thinking). Output rows {"id", "kind", "prefix", "cont", "split"} for fwd_tool.py read; `split` is
fit/held by problem id, so held-out separation (AUC) is measured on problems not used to fit the direction.

    python src/contrast.py <gen_run_dir> <out.jsonl> [--max-pairs 96] [--max-good 64]
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import prompts as P  # noqa: E402

CONT_CHARS = 800      # ~200 tokens of continuation
LOOP_LEAD = 1500      # chars into the loop before the repetitive span starts (the loop is established)


def words_with_pos(text):
    return [(m.group(), m.start()) for m in re.finditer(r"\S+", text)]


def loop_onset(text, n=8, win=300, thresh=0.9):
    """Char offset where the answer turns into repetition: the first window of `win` words in which >= thresh of the
    word 8-grams already occurred earlier; None if no such window."""
    w = words_with_pos(text)
    seen, dup = set(), []
    for i in range(len(w) - n + 1):
        g = tuple(x for x, _ in w[i:i + n])
        dup.append(g in seen)
        seen.add(g)
    for s in range(0, max(0, len(dup) - win), 25):
        if sum(dup[s:s + win]) >= thresh * win:
            return w[s][1]
    return None


def split_of(pid):
    return "fit" if int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 2 == 0 else "held"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model-dir", default="models/Qwen3.5-2B")
    ap.add_argument("--max-pairs", type=int, default=96)
    ap.add_argument("--max-good", type=int, default=64)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    items = {it["id"]: it for n in cfg["dataset"].split(",") for it in D.load(a.data_root, n)}
    rows = [json.loads(l) for l in open(os.path.join(a.run, "per_item.jsonl"))]

    def prompt(pid):
        return tok.apply_chat_template(P.messages(items[pid], "qwen"), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)

    out, stats = [], {"cut": 0, "onset_found": 0, "pairs": 0, "good": 0, "onset_frac": []}
    for r in rows:
        if not r["cut"] or stats["pairs"] >= a.max_pairs:
            continue
        stats["cut"] += 1
        t = r["text"]
        on = loop_onset(t)
        if on is None or on < CONT_CHARS + 200 or on + LOOP_LEAD + CONT_CHARS > len(t):
            continue
        stats["onset_found"] += 1
        stats["onset_frac"].append(on / len(t))
        p, sp = prompt(r["id"]), split_of(r["id"])
        good_at = on - CONT_CHARS                     # productive: the last stretch before the loop starts
        bad_at = on + LOOP_LEAD                       # repetitive: inside the established loop
        out.append({"id": r["id"], "kind": "pre_loop", "prefix": p + t[:good_at], "cont": t[good_at:on], "split": sp})
        out.append({"id": r["id"], "kind": "loop", "prefix": p + t[:bad_at], "cont": t[bad_at:bad_at + CONT_CHARS],
                    "split": sp})
        stats["pairs"] += 1
    for r in rows:
        if r["cut"] or r["outcome"] != "correct" or stats["good"] >= a.max_good or "</think>" not in r["text"]:
            continue
        think = r["text"].split("</think>")[0]
        if len(think) < 3 * CONT_CHARS:
            continue
        at = len(think) // 2
        out.append({"id": r["id"], "kind": "finished_mid", "prefix": prompt(r["id"]) + think[:at],
                    "cont": think[at:at + CONT_CHARS], "split": split_of(r["id"])})
        stats["good"] += 1
    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    fr = stats.pop("onset_frac")
    stats["median_onset_frac"] = sorted(fr)[len(fr) // 2] if fr else None
    stats["texts"] = len(out)
    stats["by_split"] = {s: sum(o["split"] == s for o in out) for s in ("fit", "held")}
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
