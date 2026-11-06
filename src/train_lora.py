"""LoRA SFT for Qwen3.5-2B on prompt/target jsonl (PLAN §7 Phase 4; gradients allowed here, this is the control).
Loss on target tokens only (the CoT inside <think>, the ```sql answer and <|im_end|>). Length-bucketed micro-batches
under a token budget, right padding (causal model: pads after the text never affect it; they are masked from the loss).
At the end the adapter is merged and the language-model tensors are written into a copy of the base safetensors, so
vLLM loads it exactly like the base model (vision tower and MTP weights untouched).

    python src/train_lora.py --data data/sft/synsql_20k.jsonl --out runs/lora_synsql [--epochs 1] [--r 16]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import edits as ED  # noqa: E402

BASE = os.path.join(ROOT, "models/Qwen3.5-2B")
ST = "model.safetensors-00001-of-00001.safetensors"
TARGETS = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|"
           r"out_proj)|mlp\.(gate|up|down)_proj)")


def encode(tok, rows):
    end = tok("<|im_end|>", add_special_tokens=False).input_ids
    out = []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False).input_ids
        t = tok(r["target"], add_special_tokens=False).input_ids + end
        out.append((p + t, [-100] * len(p) + t))
    return out


def batches(enc, budget, rng):
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    out, cur, mx = [], [], 0
    for i in order:
        n = len(enc[i][0])
        if cur and max(mx, n) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, n)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(enc, idx, pad_id):
    L = max(len(enc[i][0]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    lab = torch.full((len(idx), L), -100, dtype=torch.long)
    for j, i in enumerate(idx):
        x, y = enc[i]
        ids[j, :len(x)] = torch.tensor(x)
        lab[j, :len(y)] = torch.tensor(y)
    return ids.cuda(), lab.cuda()


def loss_fn(m, ids, lab):
    """m = (language_model, lm_head): the inner modules, which keep their LoRA layers after get_peft_model."""
    lm, head = m
    h = lm(input_ids=ids, use_cache=False).last_hidden_state
    sel = lab[:, 1:] != -100
    logits = head(h[:, :-1][sel]).float()          # logits only where there is a target token
    return torch.nn.functional.cross_entropy(logits, lab[:, 1:][sel], reduction="sum"), int(sel.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--budget", type=int, default=24000, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=BASE, help="model dir to start from (e.g. the merged v1 model for round 2)")
    a = ap.parse_args()
    base_dir = os.path.abspath(a.base)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open(a.data)]
    dbs = sorted({r["db_id"] for r in rows})
    rng.shuffle(dbs)
    held_db, held = set(), []
    for d in dbs:                                         # held-out by database
        if len(held) >= a.heldout:
            break
        held_db.add(d)
        held += [r for r in rows if r["db_id"] == d]
    train = [r for r in rows if r["db_id"] not in held_db]
    tok = AutoTokenizer.from_pretrained(base_dir)
    enc_tr, enc_ho = encode(tok, train), encode(tok, held)

    m = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16).to("cuda")
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=TARGETS, bias="none")
    m = get_peft_model(m, cfg)
    inner = m.get_base_model()
    parts = (inner.model.language_model, inner.lm_head)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    epoch_batches = batches(enc_tr, a.budget, rng)
    total_mb = int(len(epoch_batches) * a.epochs)
    plan = (epoch_batches * math.ceil(a.epochs))[:total_mb]
    steps = math.ceil(total_mb / a.accum)
    warm = max(10, steps // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    meta = {"train": len(train), "heldout": len(held), "heldout_dbs": len(held_db), "trainable_params": n_train,
            "micro_batches": total_mb, "steps": steps, "args": vars(a),
            "train_target_tokens": sum(sum(x != -100 for x in y) for _, y in enc_tr),
            "train_total_tokens": sum(len(x) for x, _ in enc_tr)}
    print(json.dumps(meta), flush=True)
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)

    def heldout_loss():
        m.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in batches(enc_ho, a.budget, random.Random(0)):
                l, k = loss_fn(parts, *collate(enc_ho, b, tok.pad_token_id))
                tot, n = tot + float(l), n + k
        m.train()
        return tot / n

    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    rec = {"step": 0, "heldout_loss": heldout_loss()}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    m.train()
    t0, step, run_l, run_n, toks = time.time(), 0, 0.0, 0, 0
    for mb, idx in enumerate(plan):
        ids, lab = collate(enc_tr, idx, tok.pad_token_id)
        l, k = loss_fn(parts, ids, lab)
        (l / k / a.accum).backward()
        run_l, run_n, toks = run_l + float(l), run_n + k, toks + ids.numel()
        if (mb + 1) % a.accum == 0 or mb == total_mb - 1:
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "loss": run_l / run_n, "lr": sched.get_last_lr()[0],
                       "tok_per_s": toks / (time.time() - t0), "elapsed_h": (time.time() - t0) / 3600}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_l, run_n = 0.0, 0
            if step % a.eval_every == 0 or step == steps:
                rec = {"step": step, "heldout_loss": heldout_loss()}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
    m.save_pretrained(os.path.join(a.out, "adapter"))

    # merge and write a vLLM-loadable checkpoint: base file with the merged language-model tensors replaced
    merged = m.merge_and_unload()
    sd = {k: v for k, v in merged.state_dict().items()}
    base = ED.load(os.path.join(base_dir, ST))
    changed = 0
    for name in list(base):
        if name in sd and ".layers." in name and "language_model" in name:
            new = sd[name].to(torch.float32).cpu().numpy()
            bits = ED.bf16_bits(new).reshape(base[name][1].shape)
            if not np.array_equal(bits, base[name][1]):
                base[name] = ("BF16", bits)
                changed += 1
    assert changed > 0, "no tensor changed after merge"
    mdir = os.path.join(ROOT, "models/lora", os.path.basename(os.path.normpath(a.out)))
    os.makedirs(mdir, exist_ok=True)
    for f in os.listdir(BASE):            # tokenizer/config files always come from the original base model
        if f != ST and not os.path.exists(os.path.join(mdir, f)):
            os.symlink(os.path.join(BASE, f), os.path.join(mdir, f))
    ED.save(os.path.join(mdir, ST), base, metadata={"format": "pt"})
    meta.update(changed_tensors=changed, merged_dir=mdir, train_hours=(time.time() - t0) / 3600)
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    print("TRAIN_DONE", json.dumps({"changed_tensors": changed, "merged_dir": mdir}), flush=True)


if __name__ == "__main__":
    main()
