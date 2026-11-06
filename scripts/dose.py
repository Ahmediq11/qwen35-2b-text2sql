"""Dose table from read passes: mean continuation log-prob change (edited - original) per text kind, held split.
    python scripts/dose.py <run> <texts.jsonl> <orig_tag> <tag>..."""
import json, sys, numpy as np
run, texts, orig, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
rows = [json.loads(l) for l in open(texts)]
def lp(tag):
    z = np.load(f"{run}/read/{tag}/logprobs.npz"); return [z[f"arr_{i}"] for i in range(len(rows))]
o = lp(orig)
print(f"{'tag':28} " + " ".join(f"{k:>14}" for k in ("loop", "pre_loop", "finished_mid")) + "   (held split, nats/token)")
for t in tags:
    e = lp(t); cells = []
    for k in ("loop", "pre_loop", "finished_mid"):
        d = [float((e[i] - o[i]).mean()) for i, r in enumerate(rows) if r["kind"] == k and r["split"] == "held"]
        cells.append(f"{np.mean(d):+14.4f}")
    print(f"{t:28} " + " ".join(cells))
