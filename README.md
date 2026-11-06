# Small model, honest numbers: improving Qwen3.5-2B at text-to-SQL

Can a 2-billion-parameter model become a useful text-to-SQL engine on modest hardware, and which of today's tools
actually moves the needle: **a gradient-free weight edit**, **LoRA distillation**, or **inference-time compute**?

This repository tests all three on the same model (Qwen3.5-2B, thinking mode) with the same execution-accuracy
harness, and reports every result: the ones that worked and the ones that didn't.

**Adapters on Hugging Face:** [v2 (best)](https://huggingface.co/Thegame1161/qwen35-2b-text2sql-lora-v2) · [v1](https://huggingface.co/Thegame1161/qwen35-2b-text2sql-lora-v1)

## Headline

| Qwen3.5-2B variant | BIRD dev | Spider dev | generated tokens / question |
|---|---|---|---|
| base, no thinking | 22.24% | 46.62% | ~80 |
| base, thinking (1 sample) | 33.46% | 64.12% | ~9,600 |
| + weight edit (loop removal) | not run on test sets* | | −28% vs base |
| LoRA v1 (SynSQL CoT), 1 sample | 25.51% | 60.15% | ~480 |
| LoRA v1 + 8 samples + execution repair + vote | 40.83% | 68.57% | ~4,900 |
| LoRA v2 (round 2, self-distilled on BIRD train), 1 sample | 29.09% | 61.41% | ~470 |
| **LoRA v2 + 8 samples + execution repair + vote** | **42.07%** | **72.44%** | **~4,700** |
| *reference:* XiYanSQL-QwenCoder-3B (trained on BIRD/Spider) | 52.19% | 85.20% | ~45 |

Execution accuracy (EX), 1,533 BIRD dev and 1,034 Spider dev questions, SQLite, the same prompt for every Qwen run
(schema DDL + 3 example rows, question, BIRD evidence). *The edit was measured on held-out practice data (below).

**Takeaway:** the best system is a cheap, distilled 2B model that samples 8 short answers, repairs the ones that fail
to execute, and votes by execution result: **+8.6 points on BIRD dev and +8.3 on Spider dev over the base model, with
about half the generated tokens.** Neither the weight edit nor LoRA alone improves accuracy, and the reasons are
instructive (below).

## The story, in four experiments

### 1. Reproducing the weight edit (Phase 0)
The starting point was MIT CSAIL's *Weight-Space Programming* result: an AI agent, with a budget of 50 forward passes
and **no gradient training**, found a rank-1 edit on two MLP matrices of Qwen3.5-0.8B that removes a "repetition
direction" and lifts GSM8K accuracy from ~14% to ~67%. I reimplemented the edit loader (numpy, bit-exact BF16 hashes),
verified the published checkpoint differs only in the two edited tensors, and re-measured it:

| GSM8K, 100 problems, thinking on | base | edited |
|---|---|---|
| accuracy | 10% | 68% |
| answers cut off at 32k tokens (loops) | 89% | 4% |

58 rescues, 0 regressions; every rescue was a formerly-looping answer. GATE 0 passed.

### 2. Discovering a SQL loop edit for Qwen3.5-2B (Phases 2–3, Track A)
With thinking on, 20% of the 2B model's BIRD answers loop until the 32k-token limit (32% of its failures). I built the
same kind of budgeted tool (`src/fwd_tool.py`, 50 passes, generate/read only) and followed the recipe without gradients:

- **Contrast sets** (`src/contrast.py`): from each looping answer, a productive span just before the loop starts and a
  repetitive span inside the loop: same question, same text up to the split point.
- **Direction** (`src/edit_build.py`): `u = mean(loop) − mean(pre-loop)` on the residual stream. Held-out AUC ≈ 1.0
  from layer 10 on.
- **Edit:** `W' = W − α · outer(u, mask64 · (u @ W))` on `mlp.down_proj` of layers 15 and 19 (64 writer neurons each,
  ~259k weights).
- **Controls:** dose response (repetitive text −0.030 nats/token vs productive −0.005 at α=6) and a norm-matched random
  orthogonal direction (no effect on text, no effect on loops in generation).

Frozen edit, confirmed on 512 practice questions from databases never used in discovery:

| | original | edited (α=6) | random control |
|---|---|---|---|
| cut-off answers | 14.1% | **3.9%** | 18.5% (discover set) |
| mean generated tokens | 7,198 | **5,160** | 8,879 |
| EX | 47.9% | 50.2% (+2.3 ± 1.8, n.s.) | 37.0% |

**Why it doesn't transfer to accuracy:** in math, the loop blocks an answer the model already knows. In SQL, the model
loops *because* it is confused: once the loop is removed, most of those questions finish with a wrong answer (only 4
of ~38 formerly-looping questions became correct). The edit is a real, causal, 28% speed-up with no accuracy cost,
not a capability gain: the same pattern the paper reported for Gemma.

### 3. LoRA distillation on short reasoning (Phase 4)
Hypothesis: the 2B model needs knowledge, and its 9,600-token reasoning is wasteful. I distilled short, structured
chain-of-thought from **SynSQL-2.5M** (execution-verified, synthetic databases), placed inside `<think>`:

- 10k examples (stratified by complexity), SQL verified by execution, **0 overlap** with test questions or databases
  (`src/sft_data.py`).
- LoRA r=16 on attention, linear-attention and MLP projections of all 24 layers (15.6M trainable parameters, 0.8%),
  1 epoch, 1.6 GPU-hours on one RTX PRO 6000 (`src/train_lora.py`).

Result: answers became **20× shorter** (~480 tokens, never cut off), but single-sample accuracy fell on BIRD dev
(33.5% → 25.5%). The short reasoning no longer double-checks BIRD's messy real schemas; most new failures are
invented column names (`no such column: …`). Lesson: distillation data must look like the target databases.

### 4. Spending the saved compute wisely (inference-time scaling)
Because one LoRA answer costs ~1/20 of a base answer, the model can afford several attempts (`src/vote_repair.py`):

1. 8 samples per question;
2. every sample whose SQL fails gets **one repair turn** with the SQLite error message;
3. the samples that execute are grouped by their result; the largest group wins (gold is never used).

| BIRD dev | EX |
|---|---|
| LoRA, 1 sample | 25.5% |
| + repair | 29.6% |
| vote@8 | 39.4% |
| **vote@8 + repair** | **40.8%** |
| oracle: any of the 8 correct | 66.1% (all questions) |

The 25-point gap between voting and the oracle shows where the next gain is: a better selector, not more samples.

### 5. Round 2: self-distillation on real BIRD databases
To close v1's domain gap, the v1 model answered each of 5,026 BIRD *train* questions (discover split) 4 times; for the
2,349 questions with at least one execution-correct answer, the **shortest correct answer** became a training target
(plus 3,000 unseen SynSQL rows). A second LoRA was trained on top of v1 (`src/r2_data.py`, 343 steps). The rule for
touching the test sets was fixed in advance: only if it beats v1 on the confirm split (it did: 54.9% vs 53.1%).

| vote@8 + repair | BIRD dev | Spider dev |
|---|---|---|
| base thinking, 1 sample | 33.46% | 64.12% |
| LoRA v1 | 40.83% | 68.57% |
| **LoRA v2** | **42.07%** | **72.44%** |

## What I would do next
- **More rounds / Spider in the loop:** round 2 used BIRD only; adding Spider train and iterating is cheap.
- **A better selector:** a small verifier, or schema-aware tie-breaking, to close the gap toward the 66% oracle.
- **Combine** the loop edit with the base model for latency-sensitive use, where its 28% token saving is free.

## Evaluation protocol and hygiene
- **Execution accuracy:** result multisets compared; order matters only when the gold query has a top-level
  `ORDER BY` (parsed with sqlglot). Read-only SQLite connections and per-query time limits. 44 unit tests on the scorer.
- **Locked test sets:** Spider dev and BIRD mini-dev were not looked at during discovery or training. The BIRD dev
  "tuning split" (BIRD dev minus the 498 mini-dev questions) is used for analysis; per-item failures are only read
  there.
- **Practice data** (Spider train + BIRD train) is split by database into `discover` and `confirm` halves; edits are
  chosen on `discover` and confirmed on `confirm`.
- **Decontamination:** SynSQL training items are dropped if their question or database name matches any test item.
- **Every step is logged** in [`explanations.md`](explanations.md) (one paragraph per decision, with numbers), and the
  edit's forward-pass budget in `runs/q2b_A/budget.json` (11 of 50 passes used).

## Repository map
| path | what |
|---|---|
| `src/data.py`, `src/prompts.py` | dataset loaders, schema text (DDL + example rows), prompts |
| `src/generate.py`, `src/score.py`, `src/sql_eval.py`, `src/failures.py` | vLLM generation, execution scoring, failure taxonomy |
| `src/edits.py` | bit-exact BF16 edit format (apply / verify / hash) |
| `src/fwd_tool.py`, `src/contrast.py`, `src/edit_build.py` | budgeted forward-pass tool, contrast sets, directions and edits |
| `src/sft_data.py`, `src/train_lora.py` | SynSQL SFT data and LoRA training (merged into a vLLM-loadable checkpoint) |
| `src/vote_repair.py` | sampling, execution repair and voting |
| `edits/q2b_L15L19_k64_a6/` | the frozen loop edit (manifest, sparse values, hashes) |
| `results/`, `runs/` | summaries and per-item outputs of every run |
| `PLAN.md`, `explanations.md` | the research plan and the full lab notebook |

## Reproduce
```bash
pip install vllm==0.33.0 transformers==5.20.2 peft accelerate sqlglot ijson   # torch 2.14 / CUDA 13
python src/generate.py --model qwen35-2b --profile think --dataset bird_dev,spider_dev --data-root data --samples 1 --out results/base_think
python src/score.py results/base_think --data-root data
python src/sft_data.py --synsql data/synsql --out data/sft/synsql_20k.jsonl --n 20000
python src/train_lora.py --data data/sft/synsql_10k.jsonl --out runs/lora_synsql10k
python src/vote_repair.py --path models/lora/lora_synsql10k --ids ids.json --dataset bird_dev,spider_dev --n 8 --out results/vote8
```
Data: Spider 1.0 and BIRD (official downloads, checksums in `data/SOURCES.md`), SynSQL-2.5M from Hugging Face.
Total for everything in this repository: about 16 hours of a rented RTX PRO 6000 (including idle time), roughly $15.

## Credits
- Weight-Space Programming: MIT CSAIL, [logs](https://github.com/RikiyaT/lm_understand-logs) and
  [edited checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming). The edit loader here is a
  reimplementation after reading their code.
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache-2.0), [SynSQL-2.5M / OmniSQL](https://arxiv.org/abs/2503.02240)
  (Apache-2.0), [Spider](https://yale-lily.github.io/spider), [BIRD](https://bird-bench.github.io/),
  [XiYan-SQL](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-3B-2504).
