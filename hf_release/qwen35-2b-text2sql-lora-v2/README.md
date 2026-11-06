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
- self-distillation
---

# Qwen3.5-2B Text-to-SQL LoRA v2 (SynSQL distillation + BIRD self-distillation)

Second-round LoRA for [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (thinking mode). **It stacks on top of
[v1](https://huggingface.co/Thegame1161/qwen35-2b-text2sql-lora-v1)**: merge v1 into the base model first, then apply
this adapter (code below).

- **v1** taught short, structured reasoning from 10k execution-verified SynSQL-2.5M chain-of-thought examples.
- **v2 (this adapter)** is rejection-sampled self-distillation on real BIRD databases: v1 answered 5,026 BIRD *train*
  questions 4 times each, and for the 2,349 questions with an execution-correct answer, the shortest correct answer
  (mean 429 tokens) became the target, mixed with 3,000 unseen SynSQL rows.

## Results (execution accuracy, SQLite; test sets used once)

| Qwen3.5-2B variant | BIRD dev (1,533) | Spider dev (1,034) | generated tokens / question |
|---|---|---|---|
| base, thinking, 1 sample | 33.46% | 64.12% | ~9,600 |
| v1, vote@8 + execution repair | 40.83% | 68.57% | ~4,900 |
| v2, 1 sample | 29.09% | 61.41% | ~470 |
| **v2, vote@8 + execution repair** | **42.07%** | **72.44%** | **~4,700** |

*vote@8 + execution repair*: 8 samples per question; each sample whose SQL fails gets one repair turn with the SQLite
error message; the executable samples are grouped by result and the largest group wins (no gold involved).
Test sets were not used for training or selection; the decision to evaluate v2 on them was conditioned in advance on
beating v1 on a held-out practice split (54.9% vs 53.1%).

**Limitations.** Use it with the sample → repair → vote loop: a single sample is still below the base model with long
thinking on BIRD. English, SQLite only. Below models trained directly on BIRD/Spider (e.g. XiYanSQL-QwenCoder-3B:
52.19% / 85.20% in the same harness).

## Training
LoRA r=16, alpha=32, dropout 0.05 on attention, linear-attention (`in_proj_qkv`, `in_proj_z`, `out_proj`) and MLP
projections of all 24 layers; AdamW lr 5e-5 cosine, 1 epoch (343 steps, 5,046 examples), loss on the answer only;
0.7 GPU-hours on one RTX PRO 6000. Held-out loss 0.253 (v1: 0.301).

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", dtype="bfloat16")
v1 = PeftModel.from_pretrained(base, "Thegame1161/qwen35-2b-text2sql-lora-v1").merge_and_unload()
model = PeftModel.from_pretrained(v1, "Thegame1161/qwen35-2b-text2sql-lora-v2").merge_and_unload()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
# save_pretrained() the merged model to serve it with vLLM
```
Prompt (user message): the schema as `CREATE TABLE` statements with 3 example rows per table, then
`Question: …`, `Evidence: …` (if any), and `Write one SQLite query that answers the question. Put the final query in a
```sql code block.` Use `enable_thinking=True`; sampling temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5.

## Credits
Base model: Qwen3.5 (Apache-2.0). Data: SynSQL-2.5M / OmniSQL (Apache-2.0, [paper](https://arxiv.org/abs/2503.02240)),
BIRD train questions and databases ([BIRD](https://bird-bench.github.io/)). Benchmarks: BIRD, [Spider](https://yale-lily.github.io/spider).
