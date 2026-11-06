"""Inference-time scaling for a cheap model: N samples, one execution-error repair round, and execution voting.

1. N samples per question (thinking on).
2. Each sample's SQL is executed (no gold involved). Samples whose SQL errors (or have no SQL) get one repair turn:
   the previous answer plus "the query failed with: <sqlite error>" as a follow-up user message.
3. Vote: group the samples whose SQL ran by their result (order-insensitive multiset of rows); the biggest group wins,
   ties go to the earliest sample. Gold is only used afterwards, for scoring.
Reports EX for: sample 0, sample 0 + repair, vote@N, vote@N + repair, and the oracle pass@N (+ repair).

    python src/vote_repair.py --path models/lora/lora_synsql10k --ids runs/q2b_A/ids_c1_confirm512.json \
        --dataset bird_train,spider_train --n 8 --out runs/vote/c1_lora8
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import prompts as P  # noqa: E402
import sql_eval as E  # noqa: E402

REPAIR = ("The query failed when it was run on the database:\n{err}\n\nCheck the schema above and write a corrected "
          "SQLite query. Put the final query in a ```sql code block.")


def run_sql(args):
    db, sql = args
    if sql is None:
        return {"ok": False, "err": "no SQL query found in the answer", "sig": None}
    try:
        rows = E.execute(db, sql, timeout_s=10.0)
        return {"ok": True, "err": None, "sig": sorted(Counter(tuple(E._norm(v) for v in r) for r in rows).items(),
                                                       key=repr)[:2000].__repr__()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": str(e)[:300], "sig": None}


def gold_of(args):
    db, gold = args
    try:
        return E.gold_result(db, gold)
    except Exception:  # noqa: BLE001
        return None


def is_correct(args):
    db, sql, g = args          # g = (gold rows, ordered) computed once in the parent, or None (gold_error)
    if sql is None or g is None:
        return False
    try:
        return E.results_match(E.execute(db, sql, timeout_s=10.0), g[0], g[1])
    except Exception:  # noqa: BLE001
        return False


def safe_map(fn, args, fallback, timeout=60, procs=32):
    """Pool map with a hard wall-clock limit per chunk: tasks are run in chunks of procs*8; a task still running
    `timeout` seconds after its chunk started (e.g. a SQLite query the progress handler cannot interrupt) gets
    `fallback`, and the pool is then replaced so the stuck worker is killed."""
    out, hung, pool = [], 0, Pool(procs)
    try:
        step = procs * 8
        for c in range(0, len(args), step):
            jobs = [pool.apply_async(fn, (x,)) for x in args[c:c + step]]
            start, bad = time.time(), 0
            for jb in jobs:
                try:
                    out.append(jb.get(timeout=max(0.1, timeout - (time.time() - start))))
                except Exception:  # noqa: BLE001  (TimeoutError or a worker error)
                    out.append(fallback)
                    bad += 1
            if bad:
                hung += bad
                pool.terminate()
                pool = Pool(procs)
        if hung:
            print(f"safe_map {fn.__name__}: {hung} task(s) hit the hard limit", flush=True)
        return out
    finally:
        pool.terminate()


def vote(cands):
    """cands: list of (sql, exec dict) in sample order -> chosen sql (None if nothing ran)."""
    ok = [(i, s, x["sig"]) for i, (s, x) in enumerate(cands) if x["ok"]]
    if not ok:
        return cands[0][0]
    cnt = Counter(sig for _, _, sig in ok)
    best = max(cnt.values())
    return next(s for _, s, sig in ok if cnt[sig] == best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-saved", action="store_true", help="rescore from <out>/gen0.json + repair.json (no GPU)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    keep = set(json.load(open(a.ids)))
    items = [it for n in a.dataset.split(",") for it in D.load(a.data_root, n) if it["id"] in keep]
    tok = AutoTokenizer.from_pretrained(a.path)
    msgs = [P.messages(it, "qwen") for it in items]
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for m in msgs]
    sp = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=a.max_tokens, seed=a.seed)
    max_prompt = max(len(tok(p).input_ids) for p in prompts)
    # room for the repair turn: original prompt + previous answer (~answer part) + error message + new answer
    saved = a.from_saved
    llm = None if saved else LLM(model=a.path, dtype="bfloat16", gpu_memory_utilization=0.9, seed=a.seed,
              max_model_len=max_prompt + 2 * a.max_tokens + 1024,
              limit_mm_per_prompt={"image": 0, "video": 0})
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if saved:
        g0 = json.load(open(os.path.join(a.out, "gen0.json")))
        assert g0["ids"] == [it["id"] for it in items]
        texts = g0["texts"]
        gen_tokens = g0.get("gen_tokens") or sum(len(tok(t).input_ids) for ts in texts for t in ts)
    else:
        outs = llm.generate(prompts, SamplingParams(n=a.n, **sp))
        gen_tokens = sum(len(s.token_ids) for o in outs for s in o.outputs)
        texts = [[s.text for s in o.outputs] for o in outs]
    json.dump({"ids": [it["id"] for it in items], "texts": texts, "gen_tokens": gen_tokens}, open(os.path.join(a.out, "gen0.json"), "w"))
    sqls = [[E.extract_sql(t, thinking=True)[0] for t in ts] for ts in texts]
    fail = {"ok": False, "err": "query did not finish (hard limit)", "sig": None}
    ex = safe_map(run_sql, [(it["db_path"], s) for it, ss in zip(items, sqls) for s in ss], fail)
    ex = [ex[i * a.n:(i + 1) * a.n] for i in range(len(items))]

    # one repair round for every sample whose SQL did not run (cut answers included: they have no SQL)
    rep_idx, rep_prompts = [], []
    for i, (it, ts) in enumerate(zip(items, texts)):
        for j, t in enumerate(ts):
            if not ex[i][j]["ok"]:
                ans = E.answer_part(t, thinking=True)[0].strip() or "(no answer)"
                m = msgs[i] + [{"role": "assistant", "content": ans},
                               {"role": "user", "content": REPAIR.format(err=ex[i][j]["err"])}]
                rep_idx.append((i, j))
                rep_prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=True))
    lim = max_prompt + 2 * a.max_tokens + 1024 - a.max_tokens
    keep_r = [k for k, p in enumerate(rep_prompts) if len(tok(p).input_ids) <= lim]
    rep_idx, rep_prompts = [rep_idx[k] for k in keep_r], [rep_prompts[k] for k in keep_r]
    if saved:
        rj = json.load(open(os.path.join(a.out, "repair.json")))
        rep_idx = [tuple(x) for x in rj["rep_idx"]]     # the saved run's repair set is the record
        rep_texts = rj["texts"]
        gen_tokens += rj.get("gen_tokens") or sum(len(tok(t).input_ids) for t in rep_texts)
    else:
        rep_out = llm.generate(rep_prompts, SamplingParams(n=1, **sp)) if rep_prompts else []
        rep_texts = [o.outputs[0].text for o in rep_out]
        rep_tok = sum(len(o.outputs[0].token_ids) for o in rep_out)
        gen_tokens += rep_tok
        json.dump({"rep_idx": rep_idx, "texts": rep_texts, "gen_tokens": rep_tok},
                  open(os.path.join(a.out, "repair.json"), "w"))
    wall = time.time() - t0
    del llm                    # free the GPU before the CPU-only scoring
    sqls_r = [list(ss) for ss in sqls]
    ex_r = [list(xs) for xs in ex]
    rep_sql = [E.extract_sql(t, thinking=True)[0] for t in rep_texts]
    rep_ex = safe_map(run_sql, [(items[i]["db_path"], s) for (i, _), s in zip(rep_idx, rep_sql)], fail)
    for (i, j), s, x in zip(rep_idx, rep_sql, rep_ex):
        if x["ok"]:
            sqls_r[i][j], ex_r[i][j] = s, x

    choices = {"sample0": [ss[0] for ss in sqls], "sample0+repair": [ss[0] for ss in sqls_r],
               f"vote@{a.n}": [vote(list(zip(ss, xs))) for ss, xs in zip(sqls, ex)],
               f"vote@{a.n}+repair": [vote(list(zip(ss, xs))) for ss, xs in zip(sqls_r, ex_r)]}
    golds = safe_map(gold_of, [(it["db_path"], it["gold_sql"]) for it in items], None, timeout=330)
    corr = {k: safe_map(is_correct, [(it["db_path"], s, g) for it, s, g in zip(items, v, golds)], False)
            for k, v in choices.items()}
    allc = safe_map(is_correct, [(it["db_path"], s, g) for it, ss, g in zip(items, sqls_r, golds) for s in ss], False)
    good = [i for i, g in enumerate(golds) if g is not None]   # gold_error items dropped, as in score.py
    n = len(good)
    res = {k: sum(v[i] for i in good) / n for k, v in corr.items()}
    res[f"oracle_pass@{a.n}+repair"] = sum(any(allc[i * a.n:(i + 1) * a.n]) for i in good) / n
    by_src = {src: {k: sum(v[i] for i in good if items[i]["source"] == src) /
                    max(1, sum(items[i]["source"] == src for i in good)) for k, v in corr.items()}
              for src in sorted({it["source"] for it in items})}
    summary = {"n": n, "samples": a.n, "EX": res, "by_source": by_src, "repairs_tried": len(rep_prompts),
               "repairs_ran": sum(x["ok"] for x in rep_ex), "gen_tokens": gen_tokens,
               "tokens_per_question": gen_tokens / len(items), "wall_s": wall, "path": a.path, "ids": a.ids}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, "per_item.jsonl"), "w") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"id": it["id"], "source": it["source"], "sqls": sqls[i], "sqls_repaired": sqls_r[i],
                                **{k: (corr[k][i] if i in good else None) for k in corr}}) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
