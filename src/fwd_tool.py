"""Budgeted forward-pass tool (PLAN §5). Every successful `generate` or `read` call costs 1 pass from
<run>/budget.json; failed calls are logged but not charged (as in the reference harness). Analysis of the saved
outputs is CPU-only and never calls the model.

    python src/fwd_tool.py --run runs/q2b_A generate --state original|edits/<name> --dataset bird_train,spider_train \
        --ids ids.json [--samples 1] [--profile think] --tag <short name>
    python src/fwd_tool.py --run runs/q2b_A read --state original|edits/<name> --texts texts.jsonl --tag <short name>

model_state = base weights (+ an edit folder with manifest.json/edit.npz in the src/edits.py format). Generation
writes an edited checkpoint once under models/edited/<edit name>/ (hash-verified) and runs vLLM on it.
`read` runs teacher-forced passes in transformers and saves, per text: mean residual stream (output of every decoder
layer) and mean MLP intermediate activation (input of down_proj) over the continuation tokens, plus the per-token
log-probs of the continuation. Texts: jsonl rows {"id", "prefix", "cont"}; at most 1,024 per pass.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import edits as ED  # noqa: E402

BASE = {"qwen35-2b": os.path.join(ROOT, "models/Qwen3.5-2B")}
ST = "model.safetensors-00001-of-00001.safetensors"
LIMITS = {"generate_problems": 512, "read_texts": 1024}


def budget(run):
    p = os.path.join(run, "budget.json")
    return json.load(open(p)) if os.path.exists(p) else {"limit": 50, "used": 0, "log": []}


def charge(run, entry, ok):
    b = budget(run)
    entry.update(ok=ok, time=datetime.datetime.utcnow().isoformat(timespec="seconds"))
    if ok:
        b["used"] += 1
        entry["pass"] = b["used"]
    b["log"].append(entry)
    json.dump(b, open(os.path.join(run, "budget.json"), "w"), indent=1)
    return entry


def materialize(model, state):
    """Edited checkpoint dir for vLLM: base files symlinked, the one safetensors file rewritten with edited tensors."""
    if state == "original":
        return BASE[model]
    name = os.path.basename(os.path.normpath(state))
    out = os.path.join(ROOT, "models/edited", name)
    done = os.path.join(out, "VERIFIED.json")
    if os.path.exists(done):
        return out
    os.makedirs(out, exist_ok=True)
    base = BASE[model]
    for f in os.listdir(base):
        if f != ST and not os.path.exists(os.path.join(out, f)):
            os.symlink(os.path.join(base, f), os.path.join(out, f))
    edited, rows = ED.apply_edit(os.path.join(base, ST), state)
    assert all(r["base_ok"] and r["edited_ok"] for r in rows), rows
    tensors = ED.load(os.path.join(base, ST))
    for n, bits in edited.items():
        tensors[n] = ("BF16", bits)
    ED.save(os.path.join(out, ST), tensors, metadata={"format": "pt"})
    json.dump({"edit": os.path.abspath(state), "rows": rows}, open(done, "w"), indent=1)
    return out


def cmd_generate(a):
    ids = json.load(open(a.ids))
    assert len(ids) <= LIMITS["generate_problems"], f"{len(ids)} problems > {LIMITS['generate_problems']}"
    out = os.path.join(a.run, "gen", a.tag)
    entry = {"type": "generate", "state": a.state, "ids": a.ids, "n": len(ids), "samples": a.samples,
             "profile": a.profile, "tag": a.tag}
    try:
        path = materialize(a.model, a.state)
        py = sys.executable
        cmd = [py, os.path.join(ROOT, "src/generate.py"), "--model", a.model, "--profile", a.profile,
               "--dataset", a.dataset, "--data-root", a.data_root, "--ids", os.path.abspath(a.ids),
               "--samples", str(a.samples), "--seed", str(a.seed), "--out", os.path.abspath(out)]
        if a.state != "original":
            cmd += ["--path", path, "--edit", os.path.abspath(a.state)]
        subprocess.run(cmd, check=True, cwd=os.path.join(ROOT, "src"))
        subprocess.run([py, os.path.join(ROOT, "src/score.py"), os.path.abspath(out), "--data-root", a.data_root], check=True,
                       cwd=os.path.join(ROOT, "src"))
        s = json.load(open(os.path.join(out, "summary.json")))
        entry["result"] = {k: s.get(k) for k in ("n", "EX", "cut_share", "mean_tokens", "gpu_hours")}
        print(json.dumps(charge(a.run, entry, True)))
    except Exception as e:
        entry["error"] = repr(e)[:400]
        print(json.dumps(charge(a.run, entry, False)))
        raise


def load_hf(model, state):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    path = BASE[model]
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    if state != "original":
        edited, rows = ED.apply_edit(os.path.join(path, ST), state)
        assert all(r["base_ok"] and r["edited_ok"] for r in rows), rows
        params = dict(m.named_parameters())
        for n, bits in edited.items():
            p = params[n]
            w = torch.from_numpy(ED.bf16_to_f32(bits).astype(np.float32)).to(p.device, torch.bfloat16)
            assert w.shape == p.shape
            p.data.copy_(w)
    return tok, m


def cmd_read(a):
    import torch
    rows = [json.loads(l) for l in open(a.texts)]
    assert len(rows) <= LIMITS["read_texts"], f"{len(rows)} texts > {LIMITS['read_texts']}"
    out = os.path.join(a.run, "read", a.tag)
    os.makedirs(out, exist_ok=True)
    entry = {"type": "read", "state": a.state, "texts": a.texts, "n": len(rows), "tag": a.tag}
    try:
        t0 = time.time()
        tok, m = load_hf(a.model, a.state)
        lm = m.model.language_model
        layers = lm.layers
        cap = {}

        def span_mean(x):
            return x[0, cap["lo"]:cap["hi"]].float().mean(0).cpu().numpy()

        hooks = []
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(
                lambda mod, inp, outp, i=i: cap["resid"].__setitem__(i, span_mean(outp[0] if isinstance(outp, tuple) else outp))))
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda mod, inp, i=i: cap["act"].__setitem__(i, span_mean(inp[0]))))
        R, A, meta, lps = [], [], [], []
        for r in rows:
            p_ids = tok(r["prefix"], add_special_tokens=False).input_ids
            c_ids = tok(r["cont"], add_special_tokens=False).input_ids
            ids = torch.tensor([p_ids + c_ids], device="cuda")
            cap.update(lo=len(p_ids), hi=len(p_ids) + len(c_ids), resid={}, act={})
            with torch.no_grad():
                h = lm(input_ids=ids, use_cache=False).last_hidden_state
                logits = m.lm_head(h[0, len(p_ids) - 1:len(p_ids) + len(c_ids) - 1]).float()
                lp = torch.log_softmax(logits, -1).gather(1, ids[0, len(p_ids):, None])[:, 0].cpu().numpy()
            R.append(np.stack([cap["resid"][i] for i in range(len(layers))]))
            A.append(np.stack([cap["act"][i] for i in range(len(layers))]))
            lps.append(lp.astype(np.float32))
            meta.append({"id": r["id"], "prefix_tokens": len(p_ids), "cont_tokens": len(c_ids),
                         "mean_lp": float(lp.mean())})
            del h, logits
        for hk in hooks:
            hk.remove()
        np.save(os.path.join(out, "resid.npy"), np.stack(R).astype(np.float32))      # [texts, layers, hidden]
        np.save(os.path.join(out, "mlp_act.npy"), np.stack(A).astype(np.float32))    # [texts, layers, intermediate]
        np.savez(os.path.join(out, "logprobs.npz"), *lps)
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
        entry["result"] = {"wall_s": round(time.time() - t0, 1),
                           "mean_lp": float(np.mean([x["mean_lp"] for x in meta]))}
        print(json.dumps(charge(a.run, entry, True)))
    except Exception as e:
        entry["error"] = repr(e)[:400]
        print(json.dumps(charge(a.run, entry, False)))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen35-2b", choices=BASE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--state", required=True)
    g.add_argument("--dataset", default="bird_train,spider_train")
    g.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    g.add_argument("--ids", required=True)
    g.add_argument("--samples", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--profile", default="think")
    g.add_argument("--tag", required=True)
    r = sub.add_parser("read")
    r.add_argument("--state", required=True)
    r.add_argument("--texts", required=True)
    r.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(a.run, exist_ok=True)
    b = budget(a.run)
    if b["used"] >= b["limit"]:
        sys.exit(f"budget spent: {b['used']}/{b['limit']} passes")
    if a.state != "original":
        assert os.path.exists(os.path.join(a.state, "manifest.json")), a.state
    {"generate": cmd_generate, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    main()
