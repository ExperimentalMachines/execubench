# Metrics: what each number means and where it comes from

Every published number in execubench has one definition, written here, and one source
artifact, named in the result file. If a figure cannot be traced to a raw file from a device
run, it is not published. This file is the contract between the harness, the schemas in
`schemas/` and anyone who cites a result. It is a plan: the harness that implements it is
built and tested in phase P1 (`docs/PLAN.md`), and any definition P1 shows to be unworkable
is changed here first.

## Provenance

Every value carries one of five provenance labels.

| Label | Meaning | Example |
|---|---|---|
| `measured` | Observed by our harness during a run on the device | decode tok/s, sampled RSS |
| `reported` | Read from the device about itself, not timed by us | `ro.soc.model`, `MemTotal`, MIDR |
| `published` | Stated by a chip or phone maker, with a URL and a retrieval date | NPU name, marketed RAM |
| `derived` | Computed from other values by a formula written next to it | theoretical DRAM bandwidth from data rate and bus width |
| `unknown` | Not found in the maker sources reviewed | memory bandwidth for every phone so far |

In the records each value is a sourced object carrying its own label (`schemas/common.schema.json`
`sourced`), so one field can hold a reported value for one phone and a published one for
another (memory type: the Pixels' bootloaders report it, the Xiaomi 13's maker publishes it).
A published table either shows the label per cell or splits the column by label; it never
shows mixed values unlabelled. Fields without a sourced object (the CPU topology block, for
example) are all `reported` from the probe. `unknown` is a legitimate value, never filled with
a third-party guess.

## Timing source

The runtime under test is ExecuTorch **1.5.1**'s `TextLLMRunner` (C++), reached from Kotlin
through the ExecuTorch Android AAR (`org.pytorch:executorch-android:1.5.1`) in ExecuServe; the
version is pinned in `config/versions.env`. Facts checked in the v1.5.1 sources
(`extension/llm/runner/text_llm_runner.cpp`, `text_token_generator.h`, `stats.h`, `util.h`,
`irunner.h`, `extension/android/jni/jni_layer_llama.cpp`; tag commit
`3b60683923245cf472b7323426920e15623ba361`). The fields used below have the same meaning at
v1.4.0 (`3dd7ccd1d863`). Between the two tags `stats.h` gained
`aggregate_model_execution_time_ms` (time inside forward passes), and the stats JSON now also
carries the existing `model_execution_start_ms` and `model_execution_end_ms`.

- `time_in_ms()` returns integer milliseconds from **`CLOCK_REALTIME`** on Android, a wall
  clock that time synchronisation may adjust mid-run. A total-duration check is not enough (two
  opposite adjustments could cancel), and a Kotlin callback is not a token: the JNI buffers
  pieces until they form valid UTF-8, so the first `onResult` can follow several sampled tokens.
  ExecuServe therefore takes **native** monotonic timestamps (docs/EXECUSERVE-CONTRACT.md,
  `native_monotonic`): at the runner call, in the native token callback for the first sampled
  token (before UTF-8 buffering), and at the runner's return. A request whose runner and native
  monotonic durations differ in either stage (TTFT, decode) by more than 5 ms plus 1 percent is
  flagged `clock_disagreement` and its timings are not published. The time between the first
  sampled token and the first Kotlin callback is buffering, recorded separately
  (`first_callback_ns`) and never counted as clock disagreement. Until the ExecuServe benchmark
  build exists, stage-clock integrity is unresolved and no timing is published.
- `inference_start_ms` is read **before** the prompt is tokenized.
- `first_token_ms` and `prompt_eval_end_ms` are two consecutive clock reads right after
  prefill returns the first sampled token and before that token is decoded to text. They are
  usually equal but not guaranteed to be.
- The JSON the runner reports names the counts `prompt_tokens` and `generated_tokens`.
  `generated_tokens` counts decode-loop steps (forward passes) **after** the token prefill
  sampled. A run that stops on an end-of-sequence token counts that step. After a separate
  prefill call, an empty-prompt `generate` reports `prompt_tokens` as the accumulated position,
  not a timed suffix; the harness does not use that path.
- Token callbacks run synchronously inside the decode loop, so the JNI hop into Kotlin is
  inside decode time: the cost a real app pays.
- **Output caps are not enforced by the stock Android API.** `LlmModule.generate(prompt,
  config, callback)` does not forward `maxNewTokens` to the runner (1.4.0 and 1.5.1 alike), and
  `seqLen` cannot stand in for it on these exports (next point). The contract requires a native
  cap (`native_max_new_tokens`).
- **The runtime does not guard the window on these exports.** Every v1 export has
  `get_max_seq_len` 2,048 below `get_max_context_len` 8,192, which sends the runner down its
  sliding-window branch: occupied positions are ignored when it resolves how many tokens may
  follow, and only the new prompt is checked against the window. With a reused KV cache the
  server must enforce capacity itself (`capacity_guard`).
- Android's `LlmGenerationConfig` has no `ignore_eos`, so the speed track cannot force a fixed
  number of decode steps; requests that stop early are excluded from speed cells with reason
  `early_stop` (below).
- A runtime upgrade changes results. Most of the 1.4.0 `.pte` files checked give different
  greedy text under the 1.4.0 and 1.5.1 runtimes on the same host
  (`data/runtime/compat-1.4.0-vs-1.5.1/SUMMARY.md`); the files repeated under one runtime gave
  identical output each time (`repeat-check/` there). The runtime version is part of every cell
  key, and results are never pooled across runtime versions.

P1 tests the adapter contract (the acceptance tests in docs/EXECUSERVE-CONTRACT.md) before any
number is published.

## Per-request timings

| Field | Definition | Unit |
|---|---|---|
| `ttft_ms` | `first_token_ms - inference_start_ms`: tokenize plus prefill, as the runner sees it | ms |
| `prefill_tokens` | `prompt_tokens`: tokens actually run through prefill. With a reused KV cache this is the new suffix only; P1 verifies this on ExecuServe's cached path | tokens |
| `cached_tokens` | Prompt tokens served from the KV cache instead of prefilled | tokens |
| `prefill_tps` | `prompt_tokens / ((prompt_eval_end_ms - inference_start_ms) / 1000)`, exactly upstream's `prefill_token_per_sec`, so it includes tokenization | tok/s |
| `decode_steps` | `generated_tokens` as defined above | steps |
| `decode_ms` | `inference_end_ms - prompt_eval_end_ms` | ms |
| `decode_tps` | `decode_steps / (decode_ms / 1000)`, upstream's `decode_token_per_sec` | tok/s |
| `sampled_tokens` | `decode_steps + 1`: tokens sampled, including special and end-of-sequence tokens, not necessarily visible text | tokens |
| `e2e_ms` | Host monotonic clock from request sent to last byte received, over adb forward | ms |
| `client_ttft_ms` | Host monotonic clock from request sent to the first streamed content byte | ms |
| `forward_ms` | `aggregate_model_execution_time_ms` (new in 1.5.1): time inside forward passes only, without sampling, tokenization or callbacks; kept in the raw runner stats, not yet a published column | ms |

Every request also records the host's monotonic `sent_ns`, `first_byte_ns` and `done_ns`, on the
same clock as `samples.jsonl`. A sample belongs to a request when its whole read, from
`t_start_ns` to `t_end_ns`, falls between that request's `sent_ns` and `done_ns`; a read that
straddles a boundary belongs to neither. Two clocks are never mixed: the phone's native
monotonic timestamps (`timings.native`) are used only to compute durations on the phone, and
samples are matched to requests only on the host's clock. No mapping between the two clocks
is assumed, so no phase inside a request (prefill versus decode) is assigned samples from host
time alone.

`execubench validate` recomputes every derived timing from its coordinates and refuses a record
where they differ: `e2e_ms` and `client_ttft_ms` from the host timestamps, the `monotonic`
durations from the four native timestamps (which must be in order), and `clock_disagreement`
from the runner and native durations. A successful record without all four native timestamps
is refused, as is a record whose `thermal_event` does not match its `thermal_status_max`.

`ttft_ms` and `client_ttft_ms` are both kept: the first is what the runtime costs, the second
adds HTTP, adb and template rendering, and the gap is itself reported.

**Validity rules.** `decode_tps` is `null` when `decode_ms < 250` or `decode_steps < 16`;
`prefill_tps` is `null` when the prefill interval is under 50 ms. Integer milliseconds carry more
than a few percent of error below those. Such requests still count for quality.

## Two kinds of speed numbers

1. **Controlled speed** (the speed track). Synthetic prompts of fixed token lengths, a fixed
   decode target, the KV cache reset before every request, after warm-up. To get exactly `N`
   decode steps a request asks for `N + 1` sampled tokens (prefill samples the first), and a
   request that stops early on end of sequence is excluded from the cell with reason
   `early_stop`. These are the headline prefill and decode figures.
2. **Task speed** (recorded on every quality request). TTFT and decode on real prompts of
   varying length. Reported only as distributions, beside the prompt-length distribution that
   produced them, never as a single "speed of the phone".

## Cells and pooling

A published distribution belongs to one **cell**, and requests are pooled only within a cell.
The cell key is:

- model file (sha256), tokenizer sha256;
- device id (model, Android version and build fingerprint hash; see `docs/DEVICES.md`);
- track, dataset or speed bucket: prompt-length target and decode-step target;
- protocol hash: fixed performance mode, cooldown thresholds, warm-up count, sampling period;
- runtime identity: ExecuTorch version, ExecuServe APK sha256, the **effective** thread count
  as ExecuServe reports it after load (the requested count, where 0 means "runtime chooses",
  is recorded but is not a key), KV-reuse mode.

Pooling across physical units of one device id is allowed and recorded (`device_units`, and
`jobs`). Pooling across any other part of the key is refused by the summariser.

## Percentiles and intervals

For any timing distribution we publish p25, p50, p75, p95 and p99 with `n`, the number of jobs,
the number of distinct units and the number of distinct days beside them.

- Method: linear interpolation between closest ranks (NumPy `method="linear"`, Hyndman and
  Fan type 7), stated in every summary file.
- **Reporting policy**, not a precision guarantee: p25, p50 and p75 need `n >= 5`; p95 needs
  `n >= 20`; p99 needs `n >= 100`. Below that the cell is `null` with reason
  `insufficient_samples`, never an extrapolation.
- Requests inside one job share a unit and a thermal history, so they are not independent.
  Uncertainty for a cell's median is a **cluster bootstrap over jobs** (2,000 resamples of
  whole jobs, percentile interval), published only when the cell has at least 5 jobs;
  otherwise the per-job medians are listed instead of an interval.
- Quality accuracy carries a Wilson 95% interval only for single-turn, binary-graded datasets
  with independent rows (GSM8K, BFCL single-turn categories, RetrievalQA exact match). For
  multi-turn conversations, IFEval's instruction-level scores and any continuous score (token
  F1), the interval is a bootstrap over conversations or prompts, the unit the score depends on.

## Memory

Sampled by the host over adb from the server process every 250 ms during a request, plus once
before and once after. Units are MiB (2^20 bytes) throughout; each sample carries its read's
host monotonic start, end and midpoint. Every read writes a sample, with the raw command output
the values were parsed from, or with a `read_error` (adb failure, or output that did not parse)
and no values; a failed read never stops the other kinds of read, and a gap is never filled.
A memory sample without a valid interval, from another job, or contradicting another read at
the same instant stops the summary instead of being placed by guesswork.

| Field | Source | Notes |
|---|---|---|
| `rss_sampled_peak_mib` | Maximum `VmRSS` among the request's samples | A **sampled** peak: a spike shorter than the sampling period can be missed. This is the per-request "peak memory" |
| `rss_mean_mib` | Time-weighted mean of sampled `VmRSS` during the request | The "average memory" column |
| `rss_anon_mib`, `rss_file_mib` | `RssAnon`, `RssFile` at the sampled peak | XNNPACK repacks weights into anonymous memory; the split shows what is resident versus mapped |
| `pss_total_mib` | `dumpsys meminfo <pid>` TOTAL PSS, once after the request | Android's own accounting, shared pages divided |
| `process_hwm_mib` | `VmHWM` at the end of the job | The process's lifetime high-water mark, which includes load and warm-up. Published **per job and model**, never per request |

Whether the adb shell may read another app's `/proc/<pid>/status` on Device Farm phones has
**not yet been checked**; P1 checks it, and every job records whether it could. If it cannot,
ExecuServe reports its own `/proc/self/status` through its status endpoint instead.

## Thermal, power and clocks

| Field | Source | Notes |
|---|---|---|
| `battery_temp_c` | `dumpsys battery` `temperature` / 10 | At request start and end. Device Farm phones are on USB power throughout, so charging heat is part of the reading. Where the thermal HAL is used instead, battery means the sensors named `BAT`, `battery`, `BATTERY` or `MAINBATRAW`; `SUBBAT` sensors reading 0 are placeholders and excluded |
| `thermal_status` | `dumpsys thermalservice` `Thermal Status` (0 none to 6 shutdown), sampled every 1 s | Sampled inside requests, not just at their boundaries; an event shorter than the sampling period can still be missed |
| `skin_temp_c`, `soc_temp_c` | Thermal HAL **current** temperatures of type SKIN (3) and CPU (0), every 1 s | The dump also prints a cached block that can be minutes old; it is never used as a reading |
| `cpu_cur_mhz` | `scaling_cur_freq` per core, every 250 ms | What the governor set, not what the core retired. Readable by the shell on all 19 probed phones |
| `cpufreq_time_in_state` | Per clock domain, before and after each block | Readability checked in P1 (probe v3 records it) |
| `cycles_per_wall_s` | `simpleperf stat -p <pid> -e cpu-cycles:u` cycles / wall seconds | **Unverified, and not a clock frequency**: it sums all of the process's threads and includes their idle time. P1 tests whether per-process user-mode counting works at all (system-wide counting failed on all 19 phones, three different ways; `docs/DEVICES.md`) |

Two separate indicators, never one "throttled" verdict:

- `thermal_event`: any sampled `thermal_status > 0` during the request.
- `clock_drop`: the **top clock domain's** median sampled `scaling_cur_freq` over the request
  (reads wholly inside its host send and receive times) is more than 15 percent below the same
  statistic for the first request of the same cell in the same job. Prefill and decode are not
  separated, because samples cannot be placed inside a request on the phone's clock. A drop can come from heat, from the governor or
  from scheduling; the indicator records the observation, not its cause.

Requests with either indicator stay in the data, are counted in the cell's `excluded` table
when a summary is computed both ways, and are never silently dropped.

## Quality

| Field | Definition |
|---|---|
| `correct` | The dataset's own grader says the answer is right (per dataset in `docs/DATASETS.md`) |
| `accuracy` | `correct / attempted`. The **eligible** rows for a model are fixed before collection by tokenizing each fully rendered prompt with that model's tokenizer: a row whose prompt plus output cap does not fit the model's window is ineligible, listed by ID and excluded from the denominator. Any eligible row that then errors, times out or overflows counts as attempted and wrong |
| `ci95` | As in "Percentiles and intervals" |
| `tool_call_valid` | The output parsed into a call with the declared schema: the name exists, arguments are JSON, types fit |
| `tool_call_correct` | The official checker accepts the call (BFCL AST checker, or the multi-turn state and response checks) |
| `tool_decision_vs_label` | RetrievalQA, search tool offered: called the tool exactly when the row's label says parametric knowledge is not enough |
| `tool_decision_vs_own` | Same, judged against the model's own closed-book result on the same device id and build: called exactly when its closed-book answer was wrong |
| `retrieval_gain` | Closed-book accuracy versus search-tool-mode accuracy on the same rows: whether retrieving actually helped |

A model that cannot attempt a dataset at all (no tool syntax in its template) gets `null` with
a reason, never zero.

## What we do not publish

- A single composite score across datasets. Per-dataset scores stay visible.
- Speed from a request that errored, timed out, stopped early in the speed track, or failed
  the clock-agreement check.
- Any figure from a job whose model file's sha256 does not match the export report.
- Memory bandwidth or BF16 TFLOPS as measured values. They are `published` or `unknown`; a
  measured microbenchmark would be a separate, explicitly named column (`docs/DEVICES.md`).
