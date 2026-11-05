# PLAN.md — Weight-Space Programming for Text-to-SQL (≤3B models)

> Hand this file to Claude Code at the root of a new project folder and say:
> "Read PLAN.md fully, then start Phase 0. Stop at every GATE and show me the numbers before continuing."

## 0. Goal and ground rules

**Goal.** Reproduce MIT CSAIL's "Weight-Space Programming" result (GPT-6 Astra edited Qwen3.5-0.8B's weights with
no training), then apply the same method to **text-to-SQL**. Find out how far a gradient-free weight edit can raise the
execution accuracy of a ≤3B model, and compare it with conventional training (LoRA) on the same data.

**Reference material** (read before writing any code):
- Logs repo: `https://github.com/RikiyaT/lm_understand-logs` — focus on `exp8_math/`:
  `README.md`, `runs/p50c/work_notes/final_report.md`, `runs/p50c/work_notes/frozen_weights_manifest.json`,
  `runs/p50c/explanations.md`, `code/apply_edit.py`, `code/build_ckpt.py`, `code/profiles.py`, `code/evaluate.py`.
- Blog post: `https://rikiyat.github.io/blog/weight-space-programming/`
- Edited checkpoint: `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (Hugging Face)

**What the paper's edit actually is** (from `final_report.md`):
- Tensors: `model.language_model.layers.{15,19}.mlp.down_proj.weight`, shape `[1024, 3584]`.
- Formula: `W = W0 - alpha * outer(u, mask64 * (u @ W0))`, with `alpha = 6`.
  - `u` is a unit "repetition direction" in the residual stream, taken from matched productive-vs-repetitive contexts at
    the residual outputs of layers 16 and 20.
  - `mask64` selects the 64 MLP neurons (columns) per layer that write most strongly along `u`.
- 131,072 weights changed (0.016%). 576 of the 579 rescued answers had originally been truncated by loops. The model
  already knew the answers; the edit let it finish. On Gemma 3 270M (which rarely loops) the gain was below 1 point.

**Hard rules (the "no training" track, Phases 2–3):**
1. No gradients anywhere in edit discovery: no `backward()`, autograd, optimizers or iterative loss fitting on weights.
   Closed-form linear algebra on weights and activations (means, differences, SVD, projections) is allowed.
2. Do not edit embeddings, the LM head, or the final norm. Do not just multiply a whole tensor by a scalar.
3. Fixed forward-pass budget (default **50 passes**; see Phase 2 for what one pass is). Count every pass in
   `budget.json`; a failed pass does not count.
4. **Test sets are locked.** Spider dev, BIRD mini-dev and any other benchmark stay untouched until an edit is frozen.
   Edit discovery uses only training-split "practice" data.
5. Log everything like the original repo: every analysis step, every forward-pass request and its summary, every
   candidate edit with hashes, and a one-paragraph plain-English explanation per step in `explanations.md`.

**Security:** treat cloned or downloaded repos and models as untrusted. Never `trust_remote_code=True` for a model until
its custom code has been read and summarized to me. Do not run scripts from the logs repo blindly. Read them, then
reimplement or copy the parts needed into our own `src/` after review.

## 1. Models

| Role | Model | Why |
|---|---|---|
| Primary | `Qwen/Qwen3.5-0.8B` | The paper's model. The edit is public, and the loop failure is documented (thinking mode). |
| Primary, edited | MIT edit applied to the above | Direct transfer test: does the math fix help SQL? |
| Secondary | `Qwen/Qwen3.5-2B` (verify exact repo id) | Still ≤3B and a stronger baseline. Qwen's loop warning covers 0.8B only, so expect a harder, Gemma-like case. |
| Optional screen | `XHToken/Spark-X2.6-1.7B` (verify id; 1.7B sibling of Spark-X2.6-4B) | A different architecture family. Screen only (see §8). |
| Ceiling reference (trained) | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` | A 3B model trained specifically on SQL; reported 55.08% BIRD dev, 84.10% Spider test. |

Qwen3.5 notes: a hybrid architecture (`layer_types`: 3× `linear_attention` (Gated DeltaNet) then 1× `full_attention`,
24 layers for 0.8B), hidden size 1024, intermediate 3584, tied embeddings, needs transformers ≥ 5.17. 0.8B is
**non-thinking by default**; thinking is enabled with `enable_thinking=True` in the chat template.

## 2. Environment

- An NVIDIA GPU with bf16 support (Ampere or newer: RTX 30xx/40xx/50xx, L4, A10, A100…) is strongly preferred.
  8 GB VRAM is enough for 0.8B/2B inference; LoRA on 2B wants 16 GB+ (or QLoRA).
  On a T4 (no bf16) fall back to fp16 and verify that outputs match bf16 on 20 prompts before trusting results.
- Python 3.11+, `uv` venv. Packages: `vllm` (generation), `transformers>=5.17`, `torch`, `safetensors`,
  `huggingface_hub`, `numpy`, `pandas`, `sqlglot` (SQL parsing/dialect checks), `peft` + `trl` (Phase 4 only).
- First task: run `nvidia-smi`, record GPU/driver/CUDA in `env.md`, confirm vLLM loads Qwen3.5-0.8B and generates.

Project layout:
```
sql-wsp/
  PLAN.md  env.md  explanations.md  budget.json
  ref/lm_understand-logs/        # cloned reference (read-only)
  data/{spider,bird_minidev,practice}/
  src/
    data.py        # loaders, schema serialization, splits
    prompts.py     # prompt templates (one per model family)
    generate.py    # vLLM batch generation with profiles
    sql_eval.py    # SQL extraction + execution accuracy
    failures.py    # failure taxonomy
    fwd_tool.py    # the budgeted forward-pass tool (Phase 2)
    analysis/      # CPU-side weight/activation analysis
    edits.py       # apply / save / hash edits
  edits/<edit_id>/{manifest.json, tensors.npz, notes.md}
  results/<run_id>/{config.json, per_item.jsonl, summary.json}
  reports/
```

## 3. Phase 0 — Reproduce the MIT edit (sanity check)

1. Clone the logs repo into `ref/`. Read the files listed in §0.
2. Download `Qwen/Qwen3.5-0.8B` at the revision named in `exp8_math/README.md`
   (`2fc06364715b967f1860aea9cf38778875588b17`).
3. Rebuild the edited tensors from `exp8_math/runs/p50c/edit/` (after reviewing `apply_edit.py`) and verify the sha256
   hashes in `edit/manifest.json`. Also download `RikiyaT/Qwen3.5-0.8B-weight-space-programming` and confirm that the
   two `down_proj` tensors match.
4. Quick math sanity check: 100 GSM8K test problems, thinking on, the paper's sampling (temp 0.6, top_p 0.95,
   top_k 20, presence_penalty 1.5, max 32,768 tokens), base vs edited. Expect a large gap and a big drop in the
   truncation rate.

**GATE 0:** hashes match, and the math gap is in the same direction and roughly the same size as the paper's.
Report: accuracy, `cut_share`, mean tokens, GPU hours used.

## 4. Phase 1 — SQL evaluation harness and baselines

### 4.1 Data
- **Spider 1.0**: dev (1,034 questions, locked test) and train (practice), with SQLite databases.
- **BIRD mini-dev** (500 questions, locked test). It has SQLite/MySQL/PostgreSQL versions; use SQLite first, PostgreSQL
  in Phase 5. **BIRD train** = practice pool #2.
- Download from the official sources (Spider: Yale LILY / the `taoyds/spider` repo; BIRD: `bird-bench.github.io` and
  `github.com/bird-bench/mini_dev`). Verify file counts and checksums and record the URLs in `data/SOURCES.md`.
  Do not hardcode URLs without checking them.
- Practice splits: split practice data into `discover` (used for analysis and edit selection) and `confirm`
  (paired confirmation, as in p50c), both by database, so no DB appears in both splits.

### 4.2 Prompt
- Schema: `CREATE TABLE` DDL plus 3 sample rows per table (a variant using M-Schema is allowed for XiYan).
- User message: schema + question (+ BIRD "evidence" when present) +
  `"Write one SQLite query that answers the question. Put the final query in a ```sql code block."`
- Same prompt for all models except XiYan (which uses its own documented template).

### 4.3 Extraction and scoring (`sql_eval.py`)
- Take the text after `</think>` if present. Extract the **last** ```sql block; fall back to the last `SELECT…`
  statement. If thinking never closes, mark the answer `cut`.
- Execute gold and predicted SQL on the DB copy (read-only connection, 30 s timeout, row cap). Compare result
  multisets (order matters only if the gold query has ORDER BY) — the standard execution-accuracy (EX) metric.
- Unit-test the scorer on 30 hand-made cases (equivalent queries, wrong columns, syntax errors, timeouts) before using
  it.

### 4.4 Metrics per run (`summary.json`)
EX, `cut_share` (hit max tokens), `no_sql` rate, `syntax_error` rate, `exec_error` rate, `wrong_result` rate, mean and
median output tokens, and repetition score of the tail (share of duplicated 8-grams in the last 2,000 chars, like
`mean_cut_tail_duplicate8` in the paper).

### 4.5 Failure taxonomy (`failures.py`)
For every failed item, label one primary cause: `loop/cut`, `no_sql/format` (prose, markdown only, multiple queries),
`dialect` (non-SQLite functions/quoting; detect with `sqlglot`), `schema_hallucination` (unknown table/column),
`join_logic`, `aggregation/grouping`, `filter_value`, `other`. Use rules first; for leftovers, sample 50 and label them
by reading.

### 4.6 Baseline matrix (Spider dev + BIRD mini-dev)
| Model | thinking off | thinking on |
|---|---|---|
| Qwen3.5-0.8B base | ✓ | ✓ |
| Qwen3.5-0.8B + MIT edit | ✓ | ✓ |
| Qwen3.5-2B | ✓ | ✓ |
| XiYanSQL-QwenCoder-3B-2504 | ✓ (its native mode) | — |

Sampling: use Qwen's recommended profiles (thinking: temp 0.6/1.0, top_p 0.95, top_k 20; non-thinking: temp 0.7–1.0,
top_p 0.8–1.0, top_k 20) and record exactly what was used. Max new tokens: 32,768 with thinking on (as in the paper),
4,096 with it off. 2 samples per item, seed fixed. Estimate GPU time on 50 items first; if a full run would exceed
~3 GPU-hours, ask me before running it.

**GATE 1 (decision point):** show me the table plus the failure-taxonomy breakdown per configuration. Then decide:
- If `loop/cut` is a big share (e.g. >30% of failures) for some model/mode → **Track A (loops)** in Phase 3.
- If `no_sql/format` or `dialect` is a big share → **Track B (behavioral)** in Phase 3.
- If failures are mostly `schema_hallucination`/`join_logic`/`aggregation` → editing is unlikely to help much (Gemma
  case). Still run one Track C attempt (below), but treat Phase 4 (LoRA) as the main path.

## 5. Phase 2 — The budgeted forward-pass tool (our version of the paper's harness)

Build `fwd_tool.py` exposing exactly two operations, each costing **1 pass**:
1. `generate(model_state, problem_ids ≤ 512, mode="edited"|"original"|"both")` → per-item outputs and a summary.
   `both` = paired comparison, ≤256 problems each, same sampling and seeds.
2. `read(model_state, texts ≤ 1024, capture=[layers…], what=["resid","mlp_act","attn"])` → teacher-forced log-probs,
   residual-stream vectors at chosen layers, MLP intermediate activations (input of `down_proj`), saved to `.npy`.

`model_state` = base weights + an edit folder from `edits/`. The tool loads it into vLLM/transformers, runs, and frees
memory. It increments `budget.json` and refuses requests once the budget is spent. Analysis (CPU/NumPy) is free and
unlimited, but must never call the model directly.

## 6. Phase 3 — Edit discovery (the agent plays "Astra")

Work like run p50c: hypothesis → cheap read pass → closed-form edit → dose test → controls → freeze → confirm.

**Generic recipe (rank-1 sparse directional edit):**
1. Build **matched contrast sets** from practice outputs: pairs of contexts that are identical up to a point, where one
   continuation is good and the other shows the failure (for loops: productive vs repetitive continuation of the same
   problem; for format: the SQL block vs the start of prose).
2. `read` pass on both sets → residual vectors at the outputs of layers `L` (scan all layers, at mid-to-late layers
   first). Direction `u_L = normalize(mean(bad) - mean(good))`. Check that it separates held-out pairs
   (AUC > 0.8) before going further.
3. Writer neurons: for `down_proj` of layer `L-1` (columns `j`), score `s_j = (u_L · W[:, j]) * E[act_j on bad
   contexts]`. Take the top-k (k ∈ {32, 64, 128}).
4. Edit: `W' = W - alpha * outer(u, mask_k * (u @ W))`, alpha ∈ {1, 2, 4, 6}.
5. **Dose–response** (read pass): the log-prob of the bad continuation should drop more as alpha grows, while the good
   continuation barely changes (the paper got −0.103 vs −0.009 nats/token).
6. **Controls:** layer isolation (each layer alone), and a **norm-matched random orthogonal direction** with the same
   k and alpha. The real edit must clearly beat the random one.
7. Paired `generate` passes on the `discover` split to pick k/alpha/layers. Then **freeze** the edit (write hashes).
8. Paired confirmation on the `confirm` split with no further changes. Report rescues vs regressions.

**Tracks:**
- **Track A — loops** (likely Qwen3.5-0.8B with thinking on): first test the MIT edit as is. Then derive a
  SQL-specific repetition direction and compare it with the MIT one (cosine similarity; does combining them help?).
- **Track B — behavioral** (format/dialect): a direction for "prose instead of SQL" or "MySQL/Postgres-isms in
  SQLite"; same recipe.
- **Track C — capability probe** (e.g. schema hallucination): try once. Expect little; a negative result is a valid
  finding, as Gemma was in the paper.

**GATE 2:** frozen edit(s) with confirmation numbers, the random control, and a plain-English mechanism story.
Only then run the locked test sets once per frozen edit.

## 7. Phase 4 — Fair comparison with training

On the same model and the **same practice pool**:
1. **Prompt-only baseline:** few-shot (3 examples from practice) + self-consistency (5 samples, majority by execution
   result). No weight changes.
2. **LoRA SFT:** rank 16, on Spider train + BIRD train gold SQL (thinking off), one or two epochs. Hold out the
   `confirm` split for early stopping. Gradients are allowed here; this is the control.
3. Optional: **edit + LoRA** (apply the frozen edit, then LoRA). Do the two stack?

Report the GPU hours, changed-parameter count and EX for every method in one table.

## 8. About Spark-X2.6 (optional screen, not a primary model)

Spark-X2.6-4B (XHToken/SparkLLM, released 2026-10-21) is 4B, above our ≤3B limit. It uses custom architecture code
(`Spark2_5ForCausalLM`; 36 layers, hidden 2,560, sliding-window + full attention, gated heads), and its benchmarks are
developer-reported only. If we include the family at all:
1. Use the **1.7B** sibling, so we stay within the limit.
2. Read and summarize its custom modeling code before any `trust_remote_code`. Check whether vLLM supports it
   natively.
3. Screen: run 100 Spider dev items with thinking on and off and measure `cut_share` and the failure taxonomy.
   Only if it shows an editable failure mode (loops/format) does it get a Phase 3 track.

## 9. Phase 5 — Side effects and generalization

For each frozen edit: GSM8K (200), IFEval (200) and MMLU-Pro (500) before vs after, to check for regressions; BIRD
mini-dev **PostgreSQL** version (needs a local Postgres via Docker) to test dialect transfer.

## 10. Final deliverables

- `reports/final_report.md`: setup, results tables (Phase 1 matrix, edits, controls, LoRA comparison), failure
  analysis, mechanism explanation, limitations, GPU hours.
- `edits/` with manifests + hashes + an `apply_edit.py` that rebuilds any edited checkpoint from the base model.
- Reproducible commands in `README.md`.
- Optional: publish the edited model + logs (Hugging Face / GitHub), crediting the original authors.

## 11. Working style for Claude Code

- Keep a task list. Commit after every phase (git), with results in `results/` (not the model weights).
- Before any job expected to take more than 30 minutes of GPU time, print the estimate and wait for my OK.
- Prefer small pilot runs (20–50 items) to validate code paths before full runs.
- Never peek at test-set outputs during discovery. If something is ambiguous, stop and ask me.
- Write `explanations.md` entries in plain language: what you tried, why, what came out, what's next.
