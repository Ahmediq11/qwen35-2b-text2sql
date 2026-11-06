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
