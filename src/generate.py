"""vLLM batch generation for text-to-SQL runs (PLAN §4.6). Writes results/<run_id>/{config.json, gen.jsonl}.

    python src/generate.py --model qwen35-0.8b --profile think --dataset spider_dev --data-root data \
        --out results/<run_id> [--ids ids.json] [--limit N] [--samples 2]

Scoring is a separate CPU step (src/score.py), so generation and evaluation can be re-run independently.
"""
import argparse
import json
import os
import subprocess
import time

import data as D
import prompts as P

MODELS = {   # key: (local dir, family, HF repo, revision)
    "qwen35-0.8b": ("models/Qwen3.5-0.8B", "qwen", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "qwen35-0.8b-wsp": ("models/Qwen3.5-0.8B-wsp", "qwen", "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
                        "d1671a178705bcab7644806d23c535517f35723b"),
    "qwen35-2b": ("models/Qwen3.5-2B", "qwen", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "xiyan-3b": ("models/XiYanSQL-QwenCoder-3B-2504", "xiyan", "XGenerationLab/XiYanSQL-QwenCoder-3B-2504",
                 "b883e58ed83f74bab037d6a7b90c4b8706d357d7"),
}
PROFILES = {
    # Qwen3.5 thinking: the paper's evaluation sampling (model card thinking setting), 32,768 new tokens
    "think": {"thinking": True, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                                             "max_tokens": 32768}},
    # Qwen3.5 non-thinking: model card non-thinking setting (same as the reference repo's qwen35 profile), 4,096 tokens
    "nothink": {"thinking": False, "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                                                "max_tokens": 4096}},
    # XiYanSQL native: its model card's sampling (temperature 0.1, top_p 0.8, 1,024 new tokens)
    "xiyan": {"thinking": None, "sampling": {"temperature": 0.1, "top_p": 0.8, "max_tokens": 1024}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--profile", required=True, choices=PROFILES)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--ids", help="json list of item ids to run (default: all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    path, family, repo, rev = MODELS[a.model]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)   # relative to project root
    prof = PROFILES[a.profile]
    items = [it for name in a.dataset.split(",") for it in D.load(a.data_root, name)]
    if a.ids:
        keep = set(json.load(open(a.ids)))
        items = [it for it in items if it["id"] in keep]
    items = items[:a.limit] if a.limit else items

    tok = AutoTokenizer.from_pretrained(path)
    kw = {} if prof["thinking"] is None else {"enable_thinking": prof["thinking"]}
    prompts = [tok.apply_chat_template(P.messages(it, family), tokenize=False, add_generation_prompt=True, **kw)
               for it in items]
    n_prompt = [len(tok(p).input_ids) for p in prompts]
    sampling = dict(prof["sampling"], seed=a.seed)
    max_len = max(n_prompt) + sampling["max_tokens"] + 64
    os.makedirs(a.out, exist_ok=True)
    cfg = {"model": a.model, "repo": repo, "revision": rev, "family": family, "profile": a.profile,
           "thinking": prof["thinking"], "sampling": sampling, "samples": a.samples, "dataset": a.dataset,
           "n_items": len(items), "dtype": a.dtype, "max_model_len": max_len, "max_prompt_tokens": max(n_prompt),
           "mean_prompt_tokens": sum(n_prompt) / len(n_prompt), "instruction": P.INSTRUCTION,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                 text=True).stdout.strip()}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    llm = LLM(model=path, dtype=a.dtype, gpu_memory_utilization=a.mem, max_model_len=max_len, seed=a.seed,
              **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if family == "qwen" else {}))
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.samples, **sampling))
    wall = time.time() - t0
    with open(os.path.join(a.out, "gen.jsonl"), "w") as f:
        for it, o, npt in zip(items, outs, n_prompt):
            for s_i, s in enumerate(o.outputs):
                f.write(json.dumps({"id": it["id"], "sample": s_i, "text": s.text, "tokens": len(s.token_ids),
                                    "finish_reason": s.finish_reason, "prompt_tokens": npt}) + "\n")
    toks = sum(len(s.token_ids) for o in outs for s in o.outputs)
    cfg.update(wall_s=wall, gen_tokens=toks, tok_per_s=toks / wall)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(json.dumps({k: cfg[k] for k in ("model", "profile", "dataset", "n_items", "wall_s", "gen_tokens",
                                          "tok_per_s")}))


if __name__ == "__main__":
    main()
