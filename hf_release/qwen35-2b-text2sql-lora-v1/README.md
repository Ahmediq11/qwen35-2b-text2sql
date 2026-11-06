---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
datasets:
- seeklhy/SynSQL-2.5M
language:
- en
tags:
- text-to-sql
- sql
- sqlite
- lora
- peft
- qwen3.5
- bird
- spider
---

# Qwen3.5-2B Text-to-SQL LoRA (v1, SynSQL short-reasoning distillation)

A LoRA adapter for [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (thinking mode) trained on 10k
execution-verified chain-of-thought examples from [SynSQL-2.5M](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M).
It teaches the model to reason **briefly** (~480 generated tokens instead of ~9,600) before writing one SQLite query.

It is meant to be used **with sampling + execution repair + voting** (see below): alone it is less accurate than the
base model on BIRD, but it is ~20× cheaper per answer, and that budget buys several attempts.

## Results (execution accuracy, SQLite)

| Qwen3.5-2B variant | BIRD dev (1,533) | Spider dev (1,034) | generated tokens / question |
|---|---|---|---|
| base, no thinking | 22.24% | 46.62% | ~80 |
| base, thinking, 1 sample | 33.46% | 64.12% | ~9,600 |
| this adapter, 1 sample | 25.51% | 60.15% | ~480 |
| this adapter, vote@8 | 39.40% | 69.25% | |
| **this adapter, vote@8 + execution repair** | **40.83%** | **68.57%** | **~4,900** |

Same prompt for every row (schema DDL + 3 example rows, question, BIRD evidence, one instruction line). Test sets were
not used for training or selection; SynSQL items whose question or database name matched any test item were removed
(0 found). For context, XiYanSQL-QwenCoder-3B (trained on BIRD/Spider train) scores 52.19% / 85.20% in the same harness.

**Limitations.** Single-sample accuracy on BIRD is *lower* than the base model: the short reasoning does not re-check
BIRD's messy real-world schemas and sometimes invents column names. Use the repair + vote loop, or prefer the base model
with thinking when you can only afford one sample. English questions, SQLite dialect only.

## Training
- Data: 10,000 SynSQL-2.5M items, stratified by complexity (Simple 20%, Moderate 35%, Complex 30%, Highly Complex
  15%), SQL executed and required to return ≥ 1 row; prompt rendered exactly like the evaluation prompt; target = the
  CoT inside `<think>…</think>` followed by a ```sql block.
- LoRA r=16, alpha=32, dropout 0.05, on q/k/v/o, linear-attention `in_proj_qkv`/`in_proj_z`/`out_proj`, and MLP
  gate/up/down of all 24 layers (15.6M trainable parameters). AdamW, lr 1e-4 cosine, 1 epoch (795 steps), loss on the
  target only. 1.6 GPU-hours on one RTX PRO 6000. Held-out loss 0.467 → 0.301.

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "Thegame1161/qwen35-2b-text2sql-lora-v1").merge_and_unload()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

prompt = f"""{schema_ddl_with_3_example_rows}

Question: {question}
Evidence: {evidence}

Write one SQLite query that answers the question. Put the final query in a ```sql code block."""
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=True)
```
Sampling used in the results: temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5. For the full
sample → execute → repair → vote loop, and the exact prompt builder, see the [GitHub project](https://github.com/Ahmediq11/qwen35-2b-text2sql) (`src/vote_repair.py`,
`src/prompts.py`). For fast serving, merge the adapter and load the merged weights in vLLM.

Code, logs and every experiment: [github.com/Ahmediq11/qwen35-2b-text2sql](https://github.com/Ahmediq11/qwen35-2b-text2sql).

## Credits
Base model: Qwen3.5 (Apache-2.0). Data: SynSQL-2.5M / OmniSQL (Apache-2.0, [paper](https://arxiv.org/abs/2503.02240)).
Benchmarks: [BIRD](https://bird-bench.github.io/), [Spider](https://yale-lily.github.io/spider).
