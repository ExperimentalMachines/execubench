# Datasets: what we run, what we dropped, and why

Status: **proposed for suite v1, not yet frozen.** Row selections are frozen in
`suites/v1/*.ids.json` before the first scored run (phase P2 in `docs/PLAN.md`), and never
re-selected after anyone has seen results. Every revision below was read from the Hub or
GitHub API on 2026-10-04; an independent review by Codex (gpt-6.1-sol) checked the same
sources and its findings were re-verified by us before being adopted.

**Every count, label total, composition figure and length statistic below is recomputed by
`python -m execubench datasets stats`** from the pinned files, with each file's sha256, into
[`data/datasets/stats.json`](../data/datasets/stats.json). No dataset rows are committed.

## The proposal we started from, and the verdict

| Proposed | Verdict | Reason in one line |
|---|---|---|
| GSM8K | **Keep** (core, single turn) | Deterministic official grader, standard in every model card |
| IFEval | **Keep** (core, single turn) | Deterministic official grader; needs per-row output budgets |
| BFCL | **Keep**, version 4 categories pinned to one upstream commit | The tool-calling column; AST categories offline, multi-turn needs the host in the loop |
| RetrievalQA | **Keep**, the expanded release, run in three modes | Brings frozen retrieved context and answerability labels; covers PopQA and FreshQA rows |
| PopQA | **Drop as a separate dataset** | 1,570 of RetrievalQA's 2,785 rows are PopQA questions; popularity can still be joined |
| FreshQA | **Drop from the core** | No frozen canonical release, answers go stale, and its grading is an LLM judge |
| (missing) | **Add Multi-IF** (core, multi turn) | Three-turn instruction following with a deterministic grader |
| (missing) | **Add BFCL multi-turn base** (core, multi turn) | Stateful tool use across turns |
| (missing) | **Add a long-context probe** (separate track) | Exports go up to 32k windows; nothing above tests them |

Single-turn coverage: GSM8K, IFEval, BFCL AST categories, RetrievalQA closed-book and
with-context. Multi-turn coverage: Multi-IF (three user turns, the model's own earlier
replies kept), BFCL multi-turn base (tool calls executed by the host between turns), and
RetrievalQA with a search tool offered (call, tool result, answer).

## Core datasets

### GSM8K

- Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) config `main`,
  split `test`, revision `740312add88f781978c0658806c59bc2815b9866`. 1,319 rows. MIT.
- Measures: grade-school arithmetic word problems, multi-step.
- Grader: the official rule from
  [`grade_school_math/dataset.py`](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py):
  take the text after `####`, strip commas, compare exactly. The prompt asks for that
  marker. A reply with no marker is wrong, and the rate of missing markers is reported
  separately, so format failure is never confused with arithmetic failure.
- Output cap: 1,024 tokens. Truncation rate reported.
- Caveat: contamination of GSM8K in pretraining corpora is documented in general
  ([GSM1k](https://arxiv.org/abs/2405.00332)); that is not evidence about any model here, and
  the fp32 source checkpoint carries the same training exposure as its export. Comparing the
  device result with that fp32 reference measures **deployment fidelity** (what export,
  quantization and the phone cost), not contamination-free task ability. Both limits are stated
  beside every GSM8K number.

### IFEval

- Source: [`google/IFEval`](https://huggingface.co/datasets/google/IFEval) split `train`
  (the evaluation set despite the name), revision `966cd89545d6b6acfd7638bc708b98261ca58e84`.
  541 rows. Apache-2.0.
- Measures: compliance with verifiable instructions (length, keywords, casing, format).
- Grader: Google's own
  [`instruction_following_eval`](https://github.com/google-research/google-research/tree/master/instruction_following_eval),
  vendored at a pinned commit, run offline on the host. We report all four official numbers:
  prompt-level and instruction-level, strict and loose. Prompt-level strict is the headline.
- Output cap: 1,280 tokens. Some prompts require 800 or more words, so a short cap would make
  them impossible. Whether 1,280 tokens is enough for each length-constrained row under each
  model's tokenizer is audited when the suite is frozen; rows a cap makes infeasible for a model
  are ineligible for that model (listed by ID), and per-row truncation is recorded.

### Multi-IF (added)

- Source: [`facebook/Multi-IF`](https://huggingface.co/datasets/facebook/Multi-IF) file
  `multiIF_20241018.csv`, revision `0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa`. 4,501
  conversations of three turns; we use the 909 English ones.
- License: **data CC-BY-NC-2.0**, checker code Apache-2.0. We publish row IDs, scores and model
  outputs, never the dataset rows themselves.
- Measures: instruction following across turns, where turn two and three add constraints
  that must hold together with the earlier ones.
- Grader: the official checker
  ([`ifeval.py`](https://github.com/facebookresearch/Multi-IF/blob/main/ifeval.py)), pinned.
- Protocol: the model's own earlier replies stay in the history (the realistic track).
  Each turn is one request with a 1,280-token cap; the KV cache is reused across turns, as an
  app would, and the `cached_tokens` field shows how much. A conversation whose history no
  longer fits the window is scored as failed from that turn on, unless it was ineligible at
  freeze time (all three prompts plus three caps exceed the window).

### BFCL (Berkeley Function Calling Leaderboard)

- Source: upstream [`ShishirPatil/gorilla`](https://github.com/ShishirPatil/gorilla) at
  commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, directory
  `berkeley-function-call-leaderboard/bfcl_eval/data/` (files named `BFCL_v4_*`). Apache-2.0.
  The Hugging Face mirror (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`, revision
  `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`) still holds v3 files; we take data and checker
  from one upstream commit so they cannot drift apart.
- Categories in the core:

  | Category | Rows upstream | Kind |
  |---|---:|---|
  | `simple_python` | 400 | single call, one function offered |
  | `multiple` | 200 | choose one function among several |
  | `parallel` | 200 | several calls of one function |
  | `parallel_multiple` | 200 | several calls across functions |
  | `irrelevance` | 240 | no offered function fits; the right answer is no call |
  | `multi_turn_base` | 200 | stateful multi-turn tool use |

  Counts recomputed from the pinned files by `execubench datasets stats`
  (`data/datasets/stats.json`, with each file's sha256).
- Grader: the official AST checker
  (`bfcl_eval/eval_checker/ast_eval/ast_checker.py`) for single-turn categories; the official
  multi-turn checker with its executable simulated APIs for `multi_turn_base`, run **on the
  Device Farm host between generations**, because tool results must be fed back before the
  next turn. Every call, simulator result and simulator state is kept in the request's
  `tool_trace`, so a multi-turn score can be re-graded offline.
- Caps: 512 tokens per assistant step. Multi-turn: at most 20 model steps per user turn,
  upstream's own `MAXIMUM_STEP_LIMIT = 20`
  (`bfcl_eval/constants/default_prompts.py` at the pinned commit); a turn that reaches it is
  forced to quit and scored as upstream scores it.
- Prompting: each model's own chat template with the tools rendered the way that template
  renders them (ExecuServe applies the template; `/apply-template` records the exact prompt).
  Tool calls are parsed on the host from the raw text, so a parse failure is visible as one.
- Naming: results are "BFCL v4 categories under the execubench harness", never an official
  leaderboard score.
- Models whose template has no tool syntax (SmolLM2, Qwen2.5-Math) get `null` with a reason.

### RetrievalQA, expanded

- Source: [`aialt/RetrievalQA`](https://huggingface.co/datasets/aialt/RetrievalQA) file
  `retrievalqa.jsonl`, revision `a6ad065f05fccdecc1f2974ff471929a39966d2f`. 2,785 rows: 1,514
  labelled answerable from parametric knowledge and 1,271 labelled as needing retrieval.
  MIT declared. Paper: Zhang, Fang and Chen, "RetrievalQA: Assessing Adaptive
  Retrieval-Augmented Generation for Short-form Open-Domain Question Answering", Findings of
  ACL 2024 ([repository](https://github.com/hyintell/RetrievalQA)). The original release
  (`zihanz/RetrievalQA`, 1,271 rows, revision `fe71a76c`) is the retrieval-needed half.
- Composition, counted by us: PopQA 1,570, TriviaQA 898, RealtimeQA 188, ToolQA 75,
  FreshQA 54. PopQA rows join back to
  [`akariasai/PopQA`](https://huggingface.co/datasets/akariasai/PopQA) by question text for
  1,568 of 1,570 rows. PopQA question texts are not unique (13,068 distinct among 14,267
  rows), so a join can match several PopQA entries; popularity is used for stratified
  reporting only where the match is unambiguous or all matches agree.
- Context: every row carries frozen retrieved passages (no retriever on the phone). Context
  length per row: median 12,211 characters, 95th percentile 19,745 (type 7), maximum 25,428.
  Whole contexts do not fit small windows, so the with-context and search-tool modes pass
  **whole passages, in the dataset's order, until a fixed character budget** would be
  exceeded: never a passage cut mid-text, and identical bytes for every model. The budget is
  set when the suite is frozen so that every rendered prompt, plus the output cap, fits an 8k
  window under every model's tokenizer (checked by tokenizing, not by counting characters).
  Each request records the sha256 of every passage given (`retained_passages`) and whether
  any gold answer string still appears in them (`answer_in_context`), so a wrong answer on a
  row whose evidence was cut away is distinguishable from a reading failure; summaries report
  accuracy on both subsets with their own denominators (`schemas/request.schema.json`,
  `schemas/summary.schema.json`).
- Three modes on the same row IDs:
  1. **Closed book.** Question only.
  2. **With context.** Question plus the budgeted passages.
  3. **Search tool offered.** One `search(query)` tool. If the model calls it, the host
     returns the budgeted passages as the tool result and asks for the answer (two turns).
- Caps: 128 tokens for an answer; in search-tool mode, one tool call and one answer turn.
- Graders: the upstream metric code
  ([`utils.py`](https://github.com/hyintell/RetrievalQA/blob/main/utils.py)), reporting its
  substring match (permissive, upstream's primary), normalized exact match and token F1 under
  their own names. Tool decisions are two separate metrics: `tool_decision_vs_label` (the
  row's label) and `tool_decision_vs_own` (call exactly when the model's own closed-book answer
  on the same device id was wrong). Neither proves retrieval helps, so `retrieval_gain`
  (search-tool accuracy minus closed-book accuracy on the same rows) is reported beside them.
  The RealtimeQA and FreshQA rows inside RetrievalQA are frozen historical questions: their
  gold answers are as of the dataset's construction, and they are labelled as such.

## Separate tracks

### Long-context probe

For each repo, on two files (the 8k export and the largest window it publishes) and on two
phones (Galaxy S25 Ultra and Pixel 10): synthetic retrieval tasks filled to 25, 50 and 90
percent of that file's window, generated deterministically from task definitions in
[NVIDIA/RULER](https://github.com/NVIDIA/RULER) (Apache-2.0) with a fixed seed: single needle,
multiple keys, and one aggregation task. Programmatic grading. This track doubles as the long
prefill speed measurement. It is a smoke test of the window, not a RULER score.

### Controlled speed

Not a dataset: fixed prompts of 128, 512, 1,024 and 2,048 tokens (and longer where the window
allows), built from a public-domain text and cut by each model's tokenizer to the exact length,
each followed by a fixed number of decode steps. See `docs/METRICS.md`.

## Considered and not in v1

| Candidate | Why not now |
|---|---|
| FreshQA | The official repository's latest commit (`7d2d368`, 2026-05-01) lists the April 21, 2026 sheet and a "next update" on May 11, 2026 that has not appeared; the sheet's CSV export held 600 rows on 2026-10-04 (500 TEST, 100 DEV, 149 false-premise; hashed in `stats.json`), with review dates already past; grading is FreshEval, an LLM judge, and no deterministic proxy grades false-premise answers. A dated snapshot track could come later with a frozen CSV |
| PopQA standalone | Covered through RetrievalQA; standalone closed-book PopQA at 135M to 4B mostly measures a floor |
| MT-Bench | Needs an LLM judge (documented position and verbosity biases); not deterministic |
| MT-Eval | Only its recollection subsets grade without a judge (10 and 28 dialogues, `stats.json`); too small to rank |
| ARC-Easy, MMLU | Would anchor against model cards, but the runner exposes no logits, so they become generation-scored multiple choice, which is not how model cards score them |
| SQuAD 2.0 | Answerable versus unanswerable is useful; RetrievalQA's with-context mode covers grounded answering in v1 |
| Perplexity and KL against fp32 | Belongs to the exporter (execupack's gate); the Android runner exposes no logits. execubench instead records greedy token agreement between the phone and the same `.pte` on an x86 host |

## Row selection and freezing

- One seeded shuffle per dataset (seed and code in the manifest); a suite size `k` is the first
  `k` IDs, so growing a suite never changes rows already scored.
- The manifest records dataset repo, revision, file path, file sha256, row count, the selected
  IDs, the prompt template version and the output cap.
- The full selection runs on each Tier A phone (one per SoC vendor), and quality is
  published **per device**, never once per model.
- An **agreement subset** of the first 40 IDs per dataset and mode runs on every other phone
  in the standard. It reports observed agreement with the same model on the same-vendor Tier A
  phone and with the x86 host run, as counts, and is never extrapolated: zero mismatches in 40
  rows still allows a disagreement rate of up to about 7 percent (one-sided 95 percent bound),
  and multi-turn rows are not independent. Phones outside Tier A get quality only on rows they
  actually ran.
