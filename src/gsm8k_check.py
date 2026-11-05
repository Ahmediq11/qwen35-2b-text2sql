"""Phase 0 math sanity check: GSM8K test problems, base vs edited Qwen3.5-0.8B, thinking on, the paper's sampling.

Runs inside WSL with vLLM. Protocol follows ref/.../exp8_math/code/evaluate.py (reimplemented, not imported):
user turn = problem + "\\n" + "Please reason step by step, and put your final answer within \\boxed{}.",
chat template with enable_thinking=True, temp 0.6 / top_p 0.95 / top_k 20 / presence_penalty 1.5, seed 0,
graded on the last \\boxed{} after the last </think> with math-verify.

    python src/gsm8k_check.py --model <dir> --tag base --n-problems 10 --max-tokens 32768 --out results/<run_id>
"""
import argparse
import json
import os
import time

SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def tail_dup8(text, chars=2000):
    """share of duplicated word 8-grams in the last `chars` characters (like mean_cut_tail_duplicate8)."""
    w = text[-chars:].split()
    grams = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def grade(pred, gold):
    from math_verify import parse, verify
    if pred is None:
        return False
    try:
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return pred.strip().replace(",", "") == gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--problems", required=True, help="jsonl with id, problem, answer")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--mem", type=float, default=0.80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(a.problems)][:a.n_problems]
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for r in rows]
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                "max_tokens": a.max_tokens, "seed": 0}
    llm = LLM(model=a.model, dtype=a.dtype, gpu_memory_utilization=a.mem, max_model_len=a.max_tokens + 2048,
              seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.samples, **sampling))
    wall = time.time() - t0

    os.makedirs(a.out, exist_ok=True)
    per, ok, cut, toks, dups = [], 0, 0, [], []
    with open(os.path.join(a.out, f"per_item_{a.tag}.jsonl"), "w") as f:
        for r, o in zip(rows, outs):
            for s_i, s in enumerate(o.outputs):
                text = s.text
                answer_part = text.rsplit("</think>", 1)[-1] if "</think>" in text else ""
                pred = last_boxed(answer_part)
                good = grade(pred, r["answer"])
                is_cut = s.finish_reason == "length"
                ok += good
                cut += is_cut
                toks.append(len(s.token_ids))
                if is_cut:
                    dups.append(tail_dup8(text))
                f.write(json.dumps({"id": r["id"], "sample": s_i, "gold": r["answer"], "pred": pred, "correct": good,
                                    "cut": is_cut, "think_closed": "</think>" in text, "tokens": len(s.token_ids),
                                    "tail_dup8": tail_dup8(text), "text": text}) + "\n")
    n = len(toks)
    summary = {"tag": a.tag, "model": a.model, "n_problems": len(rows), "samples": a.samples, "acc": ok / n,
               "cut_share": cut / n, "mean_tokens": sum(toks) / n, "tokens_sum": sum(toks),
               "mean_cut_tail_duplicate8": sum(dups) / len(dups) if dups else None, "wall_s": wall,
               "tok_per_s": sum(toks) / wall, "dtype": a.dtype, "sampling": sampling}
    json.dump(summary, open(os.path.join(a.out, f"summary_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
