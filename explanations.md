# Explanations (plain-English log, one entry per step)

## 2026-11-04 · Phase 0.1 — Read the reference run
We cloned only the `exp8_math` folder of `RikiyaT/lm_understand-logs` (commit `4d09ba6`); the full clone kept
dropping on this connection. We read the README, the final report of run p50c and the edit tools. The winning edit
touches two MLP output matrices (layers 15 and 19) and removes a "repetition" direction from 64 columns of each. The
base model was looping in 85 of its first 96 practice answers, so the edit mostly lets the model finish thinking.
Next: rebuild the edit ourselves and check the hashes.

## 2026-11-04 · Phase 0.2/0.3 — Rebuild the edit and check hashes
We reviewed `st_numpy.py` and `apply_edit.py` (pure numpy, no network, no pickle) and reimplemented them as
`src/edits.py` instead of running the reference code. Rebuilding p50c from base `Qwen/Qwen3.5-0.8B@2fc0636` gives
exactly the base and edited sha256 values in `edit/manifest.json` for both tensors (65,260 and 64,928 bf16 entries
changed, 64 columns each). The public edited checkpoint `RikiyaT/Qwen3.5-0.8B-weight-space-programming@d1671a1` has
the same 488 tensor names as the base, and only those two tensors differ, with the same hashes. So the edit is
exactly what the report says, and nothing else in the checkpoint was changed.
Next: confirm vLLM runs the model on our GPU, then a small GSM8K pilot to measure speed.

## 2026-11-04 · Phase 0.4 (pilot) — 10 GSM8K test problems, base model
vLLM 0.33 runs Qwen3.5-0.8B on the RTX 2060 SUPER in fp16 (Turing has no bf16 and no FlashAttention 2, so vLLM uses
its Triton attention backend). Inside WSL it needed a C compiler (`build-essential`) for Triton. With the paper's
thinking-on sampling, the base model got **0/10** right: all 10 answers ran to the 32,768-token limit and none closed
`</think>`. The text tails are loops ("Wait, if the ratio is 1:2, then 1/2 of 2 is 1." repeated), with a duplicated
8-gram share of 0.80 in the last 2,000 characters. This matches the paper's picture (85 of 96 practice answers cut, 93%
of MATH-500 cut). Speed is the problem: 327,680 tokens took 36 minutes (151 tokens/s). With 32k-token sequences the KV
cache only fits about 10 at a time on 8 GB, so bigger batches will not help much. A full 100-problem base vs edited
run is roughly 5M tokens, or several GPU-hours on this card.
Next: decide with the user how to run the rest of the GSM8K check (see the GATE 0 note).

## 2026-11-06 · Phase 0.4 — 100 GSM8K test problems, base vs edited (GATE 0)
The free Colab session ran out, so we moved to a rented Vast.ai box (RTX PRO 6000 Blackwell, 96 GB, bf16 like the
paper). `scripts/remote_phase0.sh` installed vLLM 0.33.0, downloaded both models at the pinned revisions and checked the
edit again on the box: the public edited checkpoint differs from the base in exactly layers 15/19 `mlp.down_proj`, and
those tensors equal our rebuild of p50c. Then the first 100 GSM8K test problems, 1 sample each, thinking on, the
paper's sampling. Result: **base 10% → edited 68%** (paper: 13.8% → 67.1%). The share of answers cut at 32,768 tokens
fell from **89% to 4%**, and the share that closed `</think>` rose from 11% to 96%. Paired: **58 rescues, 0 losses**, and
all 58 rescues were problems where the base model had run into the token limit. Base cut answers end in loops (duplicated
8-gram share 0.85). The edited model's 4 cut answers have no repeated word 8-grams in their tail (whitespace or
non-word degeneration, as the paper also saw). The edited model's remaining failures are mostly finished-but-wrong (21)
and finished without `\boxed{}` (7). So, as in the paper, the edit mostly lets the model stop looping and finish;
correct answers get longer (median 4,247 → 8,370 tokens) because the model now finishes long reasoning it used to
abandon in a loop. GPU time 0.28 h (11.4 min base, 5.5 min edited).
Next: GATE 0 passes; Phase 1 (SQL harness and baselines).

## 2026-11-06 · Phase 1.1 — Data and harness
We downloaded Spider 1.0 (Yale page → Google Drive), BIRD mini-dev and BIRD train (the official aliyuncs links) on the
GPU box. Sizes and sha256 are recorded in `data/SOURCES.md`. The counts match the official ones: Spider dev 1,034 /
train 8,659, BIRD mini-dev 500 over 11 DBs, BIRD train 9,428 over 69 DBs. The harness is `src/data.py` (loaders, DDL +
3 sample rows schema text, practice splits), `src/prompts.py` (one Qwen prompt; XiYan's own template copied verbatim
from its model card), `src/generate.py` (vLLM, pinned model revisions, Qwen model-card sampling profiles),
`src/sql_eval.py` (extraction + execution accuracy) and `src/failures.py` (rule-based taxonomy). 44 unit tests pass, 34
of them on the scorer. In the gold self-check, every gold query scored against itself is correct, except 2 slow BIRD
mini-dev gold queries (gold timeout raised to 300 s) and a few broken practice gold queries, which are excluded as
`gold_error`. Practice data is split by database into `discover` (8,960) and `confirm` (9,127).

## 2026-11-06 · Phase 1.2 — Pilot on practice data (discover split, no test items)
We ran all 7 configurations on 50 practice questions (thinking off) or 20 (thinking on), 2 samples each. The pilot
caught a scoring bug. With thinking on, Qwen's chat template opens `<think>` in the prompt, so a cut answer has no tag in
its text, and the first scorer read SQL out of the unfinished reasoning. Fixed (thinking on + no `</think>` = cut),
with a test, and rescored. Pilot EX (small samples, practice data): 0.8B off 21%, 0.8B+edit off 17%, 2B off 43%, XiYan
76% (XiYan was probably trained on Spider/BIRD train, so its practice numbers are inflated), 0.8B on 20% with **68% cut**,
0.8B+edit on 39% with **0% cut**, 2B on 66% with 5% cut. With thinking off, most 0.8B failures are made-up
columns/tables (`schema_hallucination`). With thinking on, 27 of 32 base 0.8B failures are `loop/cut`, and the MIT
edit removes them: the math edit transfers to SQL. A full-matrix estimate is several GPU-hours, mostly the base 0.8B
thinking run (answers average 22.8k tokens because it loops to the limit).
Next: ask the user before the full matrix (> 3 GPU-hours).

## 2026-11-06 · Phase 1.3 — XiYan-7B / XiYan-3B baselines (BIRD dev + Spider dev)
Model-card protocol (1 sample, temperature 0.1, DDL schema). BIRD dev: 7B 55.84% (card DDL 57.43%), 3B 52.19% (card
52.09%). Spider dev: 7B 92.07%, 3B 85.20% (likely training contamination). Tuning split = BIRD dev minus the mini-dev
questions (1,035 items, `data/bird_dev_tune_ids.json`); failure details are read only there. XiYan fails on logic, not
loops (0–1 cut), so weight editing on XiYan is parked. Goal changed to Qwen3.5-2B, to be published on GitHub/HF.

## 2026-11-06 · Phase 1.4 — Qwen3.5-2B baseline (BIRD dev + Spider dev, 1 sample)
Generic prompt (DDL + 3 rows), Qwen sampling profiles. Thinking off: BIRD dev 22.24%, Spider dev 46.62%; most failures
are made-up columns/tables (279) and joins (219) on the tuning split. Thinking on: BIRD dev 33.46%, Spider dev 64.12%,
tuning split 36.91%. **211 of 1,035 tuning items (20%) are cut at 32,768 tokens; that is 32% of all tuning failures**,
above the PLAN's 30% threshold for Track A (loops). Mean answer length 9.6k tokens. GPU time: 0.02 h off, 1.09 h on.
Next (GATE 1): Track A on 2B thinking — test the MIT-style edit recipe for a 2B repetition direction; LoRA distillation
as the comparison.
