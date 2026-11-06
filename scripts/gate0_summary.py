"""GATE 0 summary from a base vs edited GSM8K run: table, paired rescues/losses, behaviour of failures.

    python scripts/gate0_summary.py results/p0_gsm8k100
"""
import json
import os
import sys

out = sys.argv[1]
S = {t: json.load(open(os.path.join(out, f"summary_{t}.json"))) for t in ("base", "edited")}
P = {t: {(r["id"], r["sample"]): r for r in map(json.loads, open(os.path.join(out, f"per_item_{t}.jsonl"),
                                                                    encoding="utf-8"))}
     for t in ("base", "edited")}
rows = {}
for t, s in S.items():
    v = list(P[t].values())
    fails = [r for r in v if not r["correct"]]
    rows[t] = {"acc": s["acc"], "cut_share": s["cut_share"], "think_closed": sum(r["think_closed"] for r in v) / len(v),
               "mean_tokens": round(s["mean_tokens"]),
               "median_correct_tokens": sorted(r["tokens"] for r in v if r["correct"])[
                   sum(r["correct"] for r in v) // 2] if any(r["correct"] for r in v) else None,
               "cut_tail_dup8": s["mean_cut_tail_duplicate8"],
               "fail_cut": sum(r["cut"] for r in fails), "fail_finished_wrong": sum(
                   (not r["cut"]) and r["pred"] is not None for r in fails),
               "fail_finished_no_box": sum((not r["cut"]) and r["pred"] is None for r in fails),
               "gpu_hours": s["wall_s"] / 3600, "tok_per_s": round(s["tok_per_s"]), "dtype": s["dtype"]}
keys = sorted(set(P["base"]) & set(P["edited"]))
rescues = [k for k in keys if P["edited"][k]["correct"] and not P["base"][k]["correct"]]
losses = [k for k in keys if P["base"][k]["correct"] and not P["edited"][k]["correct"]]
gate = {"n": len(keys), "acc_base": S["base"]["acc"], "acc_edited": S["edited"]["acc"],
        "gain": S["edited"]["acc"] - S["base"]["acc"], "rescues": len(rescues), "losses": len(losses),
        "rescues_where_base_was_cut": sum(P["base"][k]["cut"] for k in rescues),
        "paper_gsm8k_acc": {"unedited": 0.138, "edited_p50c": 0.671},
        "gpu_hours_total": sum(r["gpu_hours"] for r in rows.values())}
res = {"table": rows, "gate0": gate}
json.dump(res, open(os.path.join(out, "gate0_summary.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
