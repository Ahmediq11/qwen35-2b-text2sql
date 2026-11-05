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
