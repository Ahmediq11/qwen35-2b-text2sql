"""CPU analysis of read passes and construction of rank-1 sparse directional edits (PLAN §6). Never calls the model.

    python src/edit_build.py dirs  <read_dir> <texts.jsonl> <base.safetensors> <out_dirs.npz>
        per layer L: u_L = normalize(mean resid[loop] - mean resid[pre_loop]) on the fit split; held-out AUC of the
        projection (loop vs pre_loop, loop vs finished_mid); writer score of down_proj column j in layer L:
        s_j = (u_L . W[:, j]) * (E[act_j | loop] - E[act_j | pre_loop]).
    python src/edit_build.py build <dirs.npz> <base.safetensors> <edit_dir> --layers 15,19 --k 64 --alpha 2
        [--random-control <edit_dir_of_real_edit>]
        W' = W - alpha * outer(u, mask_k * (u @ W)) on model.language_model.layers.L.mlp.down_proj.weight.
        The random control uses a random unit direction orthogonal to u, the same columns, and an alpha rescaled so
        that ||W' - W||_F matches the real edit layer by layer.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edits as ED  # noqa: E402

NAME = "model.language_model.layers.{}.mlp.down_proj.weight"


def auc(pos, neg):
    """P(score(pos) > score(neg)), ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def cmd_dirs(a):
    R = np.load(os.path.join(a.read_dir, "resid.npy"))       # [n, L, d]
    A = np.load(os.path.join(a.read_dir, "mlp_act.npy"))     # [n, L, f]
    rows = [json.loads(l) for l in open(a.texts)]
    meta = json.load(open(os.path.join(a.read_dir, "meta.json")))
    assert [m["id"] for m in meta] == [r["id"] for r in rows]
    kind = np.array([r["kind"] for r in rows])
    split = np.array([r["split"] for r in rows])
    sel = lambda k, s: (kind == k) & (split == s)  # noqa: E731
    L = R.shape[1]
    W = ED.load(a.base, [NAME.format(l) for l in range(L)])
    U, S, report = np.zeros((L, R.shape[2]), np.float32), np.zeros((L, A.shape[2]), np.float32), []
    for l in range(L):
        diff = R[sel("loop", "fit"), l].mean(0) - R[sel("pre_loop", "fit"), l].mean(0)
        u = diff / np.linalg.norm(diff)
        proj = R[:, l] @ u
        w = ED.bf16_to_f32(W[NAME.format(l)][1])              # [d, f]
        dact = A[sel("loop", "fit"), l].mean(0) - A[sel("pre_loop", "fit"), l].mean(0)
        s = (u @ w) * dact
        U[l], S[l] = u, s
        top = np.argsort(-s)
        pos = s[s > 0].sum()
        report.append({"layer": l, "diff_norm": float(np.linalg.norm(diff)),
                       "auc_held_loop_vs_preloop": auc(proj[sel("loop", "held")], proj[sel("pre_loop", "held")]),
                       "auc_held_loop_vs_finished": auc(proj[sel("loop", "held")], proj[sel("finished_mid", "held")]),
                       "top64_share_of_positive_writing": float(s[top[:64]].sum() / pos) if pos > 0 else None,
                       "top64": top[:64].tolist()})
    np.savez(a.out, u=U, score=S)
    json.dump(report, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(f"{'L':>3} {'|diff|':>8} {'AUC loop/pre':>13} {'AUC loop/fin':>13} {'top64 share':>12}")
    for r in report:
        print(f"{r['layer']:>3} {r['diff_norm']:8.2f} {r['auc_held_loop_vs_preloop']:13.3f} "
              f"{r['auc_held_loop_vs_finished']:13.3f} {r['top64_share_of_positive_writing'] or 0:12.3f}")


def edited_bits(w_bits, u, cols, alpha):
    w = ED.bf16_to_f32(w_bits).astype(np.float64)
    proj = u.astype(np.float64) @ w
    mask = np.zeros(w.shape[1])
    mask[cols] = 1.0
    new = w - alpha * np.outer(u, mask * proj)
    return ED.bf16_bits(new.astype(np.float32)).reshape(w_bits.shape), float(np.linalg.norm(new - w))


def cmd_build(a):
    d = np.load(a.dirs)
    layers = [int(x) for x in a.layers.split(",")]
    base = ED.load(a.base, [NAME.format(l) for l in layers])
    real = json.load(open(os.path.join(a.random_control, "manifest.json"))) if a.random_control else None
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.edit_dir, exist_ok=True)
    z, tensors, info = {}, [], []
    for i, l in enumerate(layers):
        name = NAME.format(l)
        dt, b = base[name]
        u = d["u"][l].astype(np.float64)
        if real is None:
            cols = np.argsort(-d["score"][l])[:a.k]
            alpha = a.alpha
        else:
            ri = next(x for x in real["edit"]["layers"] if x["layer"] == l)
            cols = np.array(ri["columns"])
            r = rng.standard_normal(u.shape)
            r -= (r @ u) * u
            u = r / np.linalg.norm(r)
            _, n1 = edited_bits(b, u, cols, 1.0)                # ||dW|| is linear in alpha
            alpha = ri["dW_fro"] / n1
        e, fro = edited_bits(b, u, cols, alpha)
        flat_b, flat_e = b.reshape(-1), e.reshape(-1)
        idx = np.nonzero(flat_b != flat_e)[0]
        z[f"t{i}_idx"], z[f"t{i}_val"] = idx.astype(np.int64), flat_e[idx]
        tensors.append({"name": name, "shape": list(b.shape), "form": "sparse", "base_sha256": ED.sha256(b),
                        "edited_sha256": ED.sha256(e)})
        w = ED.bf16_to_f32(b)
        info.append({"layer": l, "alpha": float(alpha), "k": int(len(cols)), "columns": sorted(int(c) for c in cols),
                     "changed_entries": int(len(idx)), "dW_fro": fro, "rel_change": fro / float(np.linalg.norm(w)),
                     "u_sha256": ED.sha256(u.astype(np.float32))})
    np.savez(os.path.join(a.edit_dir, "edit.npz"), **z)
    man = {"tensors": tensors, "edit": {"formula": "W' = W - alpha * outer(u, mask_k * (u @ W))",
                                        "kind": "random_control" if real else "directional", "dirs": a.dirs,
                                        "layers": info}}
    json.dump(man, open(os.path.join(a.edit_dir, "manifest.json"), "w"), indent=1)
    _, rows = ED.apply_edit(a.base, a.edit_dir)
    assert all(r["base_ok"] and r["edited_ok"] for r in rows)
    print(json.dumps([{k: x[k] for k in ("layer", "alpha", "k", "changed_entries", "rel_change")} for x in info]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dirs")
    p.add_argument("read_dir")
    p.add_argument("texts")
    p.add_argument("base")
    p.add_argument("out")
    b = sub.add_parser("build")
    b.add_argument("dirs")
    b.add_argument("base")
    b.add_argument("edit_dir")
    b.add_argument("--layers", required=True)
    b.add_argument("--k", type=int, default=64)
    b.add_argument("--alpha", type=float, default=1.0)
    b.add_argument("--random-control")
    b.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"dirs": cmd_dirs, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
