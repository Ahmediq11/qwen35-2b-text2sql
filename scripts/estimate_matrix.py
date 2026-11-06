"""Estimate full baseline-matrix GPU time from the pilot runs (PLAN §4.6).
Full run per config = (Spider dev 1,034 + BIRD mini-dev 500) items x 2 samples; startup is measured per config.
    python scripts/estimate_matrix.py results/p1_pilot"""
import glob
import json
import os
import sys

FULL_ANSWERS = (1034 + 500) * 2
rows, total = [], 0.0
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*", "config.json"))):
    c = json.load(open(d))
    s = json.load(open(os.path.join(os.path.dirname(d), "summary.json")))
    answers = c["n_items"] * c["samples"]
    tpa = c["gen_tokens"] / answers
    startup_h = 5 / 60          # measured: about 4.5-5 min per config (load, torch.compile, CUDA graphs)
    gen_h = FULL_ANSWERS * tpa / c["tok_per_s"] / 3600
    total += startup_h + gen_h
    rows.append((c["model"], c["profile"], answers, s["EX"], s["cut_share"], round(tpa), round(c["tok_per_s"]),
                 round(c["wall_s"] / 60, 1), round(startup_h + gen_h, 2)))
print(f"{'model':16} {'profile':8} {'ans':>4} {'EX':>5} {'cut':>5} {'tok/ans':>7} {'tok/s':>6} {'gen_min':>7} {'est_full_h':>10}")
for r in rows:
    print(f"{r[0]:16} {r[1]:8} {r[2]:4} {r[3]:5.2f} {r[4]:5.2f} {r[5]:7} {r[6]:6} {r[7]:7} {r[8]:10}")
print(f"estimated full matrix: {total:.2f} GPU-hours")
