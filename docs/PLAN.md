# execubench plan

Benchmarks for compiled ExecuTorch models on real phones in AWS Device Farm: speed, memory,
heat and answer quality, single turn and multi turn, with every number traceable to a raw file
from a device run. Written 2026-10-04. This file holds the decisions, the evidence behind them,
the phases and the open questions; update it when any of those change.

Companion documents: `docs/METRICS.md` (definitions), `docs/DATASETS.md` (what we run and
why), `docs/DEVICES.md` (the phone standard and what the phones report about themselves).
Numbers in this file name their source file or are labelled **estimate**.

## 1. What we will claim, and what we will not

We intend to be cited. That sets the bar:

- **Claim:** for a named `.pte` (repo, revision, sha256) on a named phone model at a named
  Android build, under a named protocol, these were the measured distributions and scores,
  with `n`, the jobs, units and days behind them, and the raw logs.
- **Do not claim:** that a phone model "is" a speed. Device Farm hands out any free unit of a
  model and the units differ (Section 6.6). We publish the spread, not a single figure.
- **Do not claim:** steady-state clocks. Device Farm phones are not rooted, so we cannot pin
  CPU frequencies or disable thermal management. PyTorch removed ExecuTorch's own Device
  Farm benchmark jobs for exactly this reason
  ([pytorch/executorch#15433](https://github.com/pytorch/executorch/pull/15433): "performance
  data being unreliable on non-rooted devices"). Our answer is not to pretend otherwise but to
  measure and publish the conditions (clock samples, thermal status, temperatures, unit hash,
  build) beside every number, mark requests with thermal events or clock drops, and repeat
  across units and days.
- **Do not publish** a number whose model hash, protocol, runtime identity or device state is
  missing (`schemas/request.schema.json` refuses a successful request without them).

## 2. Principles

1. **Raw first.** The device and the test host only collect. Parsing, grading and
   summarising happen afterwards on our machine, from the artifacts, so a grader fix never
   needs another device run.
2. **Provenance on every value** (`measured`, `reported`, `published`, `derived`, `unknown`;
   `docs/METRICS.md`). Device Farm's own catalogue is never a spec source: in its 2026-10-04
   snapshot (`data/devices/catalogue/2026-10-04.json`) the Android `memory` field ranges from
   16 to 512 GB, which is storage, and `cpu.clock` is the first cluster's clock (it lists the
   Pixel 10's Tensor G5 at 2,246 MHz, the Cortex-A520 pair; the X4 core runs at 3,782 MHz).
3. **Pin everything.** Model files by sha256, datasets by revision and file hash, graders
   by commit, the harness by commit, the server APK by hash, the ExecuTorch runtime by
   version. The Hub moves under us (Section 3.1), so P2 freezes a model manifest.
4. **Grid before results.** Before any result is reported, the full grid (model by window by
   device by track) is written down with every empty cell marked. Partial coverage is never
   presented as a finding. Fill breadth first; pull artifacts continuously.
5. **No composite score.** Per-dataset and per-metric results stay visible.
6. **Independent review.** Every document and every published table goes through Codex
   (`gpt-6.1-sol`, medium effort) before it is published; dispositions are logged in
   Section 12.

## 3. What is under test

### 3.1 Models

All XNNPACK exports published by
[ExperimentalMachines/execupack](https://github.com/ExperimentalMachines/execupack) under
the `experimentalmachines` Hugging Face org, pinned in `data/models/xnnpack.json`
(`execubench models scan`). Scan of 2026-10-03T23:23Z (`scanned_utc` in that file): **71
`.pte` files in 16 repos**, all exported with ExecuTorch 1.4.0 (export provenance, kept as
recorded; the benchmark runtime is 1.5.1, Section 3.4), 32 round-to-nearest (`8da4w`) and 39
GPTQ-solved (`8da4w-gptq`), int8 per-channel embeddings, group size 32, fp32 KV cache, prefill
chunk 2,048 (the `recipe` field). For every file the Hub's LFS sha256 equals the sha256 its
export report recorded, and every tokenizer is hashed.

**The inventory moves.** A preliminary scan 70 minutes earlier (`data/models/scan-preliminary-2026-10-03.json`,
22:13Z) found 68 files (44 round-to-nearest, 24 GPTQ). In between, execupack's CI gave 11 of
the 16 repos a new revision, and three of them (Qwen2.5-0.5B, Qwen3-0.6B, SmolLM2-135M) went
from round-to-nearest to GPTQ and gained a 32k window: 68 less 12 round-to-nearest files plus
15 GPTQ files is 71. So a result names the file's sha256, never just a repo or a model name,
and P2 freezes a **model manifest** (repo, revision, file, sha256) that the v1 grid runs
against, whatever the Hub holds by then.

It moved again on 2026-10-04. The scan at 03:38Z (`data/models/xnnpack.rejected.json`) found 79
files in the same 16 repos (70 GPTQ, 4 round-to-nearest, 5 fp32): 28 files of the accepted
inventory are gone from the Hub's head, among them 7 of the 16 v1 8k files, including the
pilot's `SmolLM2-360M-Instruct-8da4w-8k.pte` (that repo now publishes fp32 only); 36 files are
new; the 43 files in both have unchanged sha256. `models scan` refused to replace the accepted
inventory because files it lists went missing (`models.problems`), so `xnnpack.json` still
holds the 2026-10-03 pins. Every pinned revision still downloads with its recorded sha256
(checked for both pilot models on 2026-10-04), so the pins, the compatibility runs and the
pilot stay valid; whether v1 moves to the new exports is an owner decision before P2.

| Family | Repos | Windows published at the 23:23Z scan |
|---|---|---|
| LFM2.5 | 1.2B-Instruct, 2.6B, and the two `heretic` (abliterated) variants | 2k to 32k |
| Qwen3 | 0.6B (to 32k), 1.7B, 4B, 4B-Instruct-2507 | 2k to 16k |
| Qwen2.5 | 0.5B (to 32k), 1.5B, 3B Instruct, Math-1.5B-Instruct (2k, 8k, 16k) | 2k to 16k |
| Llama 3.2 | 1B (to 32k), 3B Instruct | 2k to 16k |
| SmolLM2 | 135M (to 32k), 360M Instruct | 2k to 16k |

Excluded: `experimentalmachines/QwenGrad-Qwen3.5-2B-M1-DPO-v2-ExecuTorch-CPU` (no export
report, so no provenance), and `alpharomercoma/Qwen3-1.7B-ExecuTorch` (a personal copy;
the org repo is canonical). The heretic variants are benchmarked as their own models and
labelled as abliterated.

Only XNNPACK (CPU) is in scope for v1. Vulkan, QNN and MediaTek exports exist for some
models; QNN and MediaTek ones carry only a structural check (never executed). The probe shows
which vendor NPU runtimes each phone carries (`docs/DEVICES.md`), the starting point for a v2
backend track.

### 3.2 Window policy

The window changes only the KV cache and masks, not the weights. v1 runs:

- **Quality and agreement tracks at 8k** for every model: an 8k file exists in all 16 repos
  (`tests/test_models.py` checks it), and multi-turn conversations plus RetrievalQA context
  need more than 4k.
- **Speed track at 8k**, plus a **window sweep** over every published window of three models
  (SmolLM2-360M, Qwen3-1.7B, LFM2.5-1.2B: 13 files) on the Galaxy S25 Ultra and the Pixel 10,
  to measure whether the window alone moves prefill, decode or memory. The hypothesis is that
  decode speed does not depend on the window because ExecuTorch's custom SDPA attends only up
  to the current position; memory certainly does.

Estimated resident memory in GiB, from execupack's sizing model (`resident_estimate_bytes` in
`data/models/xnnpack.json`: `.pte` plus fp32 KV cache plus overhead). An **estimate**, to be
replaced by measured `rss_sampled_peak_mib`; windows a repo did not publish are shown because
the sizing model covers them.

| Model | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 0.65 | 0.74 | 0.92 | 1.28 | 2.00 |
| SmolLM2-360M | 0.86 | 1.02 | 1.33 | 1.97 | 3.23 |
| Qwen2.5-0.5B | 0.90 | 0.95 | 1.05 | 1.25 | 1.64 |
| Qwen3-0.6B | 1.37 | 1.81 | 2.69 | 4.46 | 7.99 |
| LFM2.5-1.2B | 1.44 | 1.49 | 1.59 | 1.78 | 2.17 |
| Llama-3.2-1B | 1.49 | 1.62 | 1.87 | 2.38 | 3.40 |
| Qwen2.5-1.5B / Math-1.5B | 1.62 | 1.73 | 1.96 | 2.41 | 3.32 |
| Qwen3-1.7B | 2.11 | 2.55 | 3.43 | 5.20 | 8.73 |
| LFM2.5-2.6B | 2.13 | 2.19 | 2.32 | 2.58 | 3.10 |
| Qwen2.5-3B | 2.54 | 2.68 | 2.97 | 3.55 | 4.70 |
| Llama-3.2-3B | 2.98 | 3.42 | 4.30 | 6.07 | 9.60 |
| Qwen3-4B / 4B-Instruct-2507 | 3.53 | 4.09 | 5.23 | 7.49 | 12.02 |

On the 4 GB Galaxy A17 (3.37 GiB `MemTotal`) only models estimated well under that at 8k are
plausible, and that phone is in the set precisely to find the floor; which models load there
is measured, not assumed.

### 3.3 Runtime

The phone runs **ExecuServe**
([ExperimentalMachines/execuserve](https://github.com/ExperimentalMachines/execuserve)): an
Android foreground service that loads `.pte` files through the ExecuTorch AAR (`LlmModule`, the
C++ `TextLLMRunner` underneath) and answers the OpenAI chat API, with each family's chat
template tested byte for byte against the model's own. Reasons:

- **Multi-turn needs the host in the loop.** BFCL multi-turn executes tool calls in Python
  simulators between generations, and RetrievalQA's tool mode returns a search result. A
  server on the phone and a harness on the Device Farm host make that a normal HTTP loop over
  `adb forward`.
- **It is the path a real app takes:** the same AAR, JNI and template code a user's app would
  run, not a lab binary.
- **It already reports the runner's stats** (`timings` per response) and exposes
  `/apply-template`, which gives the exact prompt bytes for every request.

ExecuServe today uses the 1.4.0 AAR and cannot serve a benchmark as it stands. What it must
provide, why, and the acceptance tests a build must pass are in
**`docs/EXECUSERVE-CONTRACT.md`** (contract 1): the 1.5.1 AAR in a benchmark flavour with a
pinned APK hash; per-request cache control; a **native** output cap, because the stock Android
`generate` does not forward `maxNewTokens`; its own **context-capacity guard**, because on these
exports the runtime's sliding-window branch ignores occupied positions; the raw runner stats;
**native** monotonic timestamps, because JNI's UTF-8 buffering makes the first Kotlin callback
an unreliable first-token time; the rendered prompt; the effective thread count; and
`/proc/self/status` memory on its status endpoint. The harness refuses to run against a build
that does not advertise the contract (`execubench/harness/run.py`).

The alternative considered was an instrumentation test APK in the style of OpenWeights'
`ExecuTorchBenchmarkEval` (no server, results written to app storage). It is simpler for
single-turn rows but cannot run host-side tool simulators mid-conversation without
reimplementing them on the phone, so it was rejected for v1.

### 3.4 Runtime version: ExecuTorch 1.5.1

The benchmark runtime is **ExecuTorch 1.5.1** (`config/versions.env`): the phone runs
`org.pytorch:executorch-android:1.5.1` (Maven has no 1.5.0), and host runs use the PyPI wheel
`executorch==1.5.1` with `torch==2.14.0`, the torch its `torch_pin.py` names (the wheel imports
torch without declaring it; `requirements/host.lock` pins both with hashes). What moving from
1.4.0 means:

- **Timing semantics are unchanged** for every field `docs/METRICS.md` uses (checked in the
  v1.5.1 sources); 1.5.1 adds `aggregate_model_execution_time_ms`, kept in the raw stats.
- **The exports stay 1.4.0 exports.** Upstream's compatibility policy
  ([`runtime/COMPATIBILITY.md`](https://github.com/pytorch/executorch/blob/v1.5.1/runtime/COMPATIBILITY.md))
  promises that a program from one release loads and executes on the next non-patch release when
  it was made with stable, non-experimental APIs; custom operators have no explicit guarantee,
  and nothing in the policy promises identical outputs. So whether a
  given file runs on 1.5.1 is checked, not assumed: `python -m execubench host run` and
  `host compare` load each file under both runtimes on this machine and diff the greedy output
  piece by piece (`data/runtime/compat-1.4.0-vs-1.5.1/`, results in `SUMMARY.md` there).
- **Outputs change with the runtime.** The same 1.4.0 file gives different greedy text under
  1.4.0 and 1.5.1 for most files checked (`SUMMARY.md` in the compatibility folder lists every
  v1 file). Within one runtime, repeated runs of a file gave identical output in the files
  checked twice (`repeat-check/` there); that is evidence for those files on this host, not a
  general property.
  XNNPACK moved to a new commit between the tags (`1adaa7c` to `92a7ad5`), with custom SDPA and
  weight-repacking changes beside it; which change causes a given difference has not been
  isolated. Consequences: the runtime version is part of every cell key, 1.4.0 and 1.5.1 results
  are never pooled, and OpenWeights-era 1.4.0 measurements are not comparable baselines.
- **Re-exporting with 1.5.1** (an execupack change) would produce new files and hashes, not a
  correction of these; until it happens the v1 grid runs the 1.4.0 exports on the 1.5.1 runtime
  and records both versions (`model.export_executorch`, `runtime.executorch`).

## 4. Device standard

Full detail, probe evidence and the published-spec table: `docs/DEVICES.md`; the tiers as data:
`data/devices/standard.yaml`. The previous standard (Galaxy S25 Ultra, Pixel 10 family, Galaxy
Tab S10+, Galaxy Z Flip7) came from Firebase Device Streaming. On Device Farm, every Android
model in the 2026-10-04 catalogue was probed (`data/devices/probe/`):

- Snapdragon 8 Elite (Galaxy S25 Ultra): **available**.
- Tensor G5 (Pixel 10, 10 Pro, 10 Pro XL): **available**.
- Dimensity 9300+ (Galaxy Tab S10+): **not in the catalogue**. The Galaxy Tab S11 (Dimensity
  9400+) is the only MediaTek chip offered with Cortex-X cores (from MIDR).
- Exynos 2500 (Galaxy Z Flip7): **not in the catalogue**. The only Exynos phone with Armv9
  cores is the Galaxy A56 (Exynos 1580, mid-range); the only flagship-class Exynos is the 2020
  Galaxy Note20 (platform `universal990`, Android 11).

Proposed v1 standard: **Tier A**, a vendor coverage set where the full quality selection runs
(Galaxy S25 Ultra, Pixel 10, Galaxy Tab S11, and the mid-range Galaxy A56 for Exynos, its tier
mismatch stated in every table). **Tier A+**, current generation: Galaxy S26 Ultra and Pixel
11. **Tier B**, reach: Galaxy A36, Redmi Note 13 Pro+ 5G, Galaxy A17. **Tier C**, anchors to
earlier work: Galaxy S24 Ultra and Pixel 9.

Each device is pinned to one Android version in the device pool. A device **id** includes a
hash of the build fingerprint, because units of one model can run different firmware (two of
19 models did across the first two probes, and the Pixel 2 XL did across the two script
checks); results are never pooled across ids.

## 5. Datasets

Full detail: `docs/DATASETS.md`; every count recomputed in `data/datasets/stats.json`.
Verdict on the proposed list: keep GSM8K, IFEval, BFCL (version 4 categories from one
upstream commit) and RetrievalQA (the 2,785-row expanded release, in closed-book,
with-context and search-tool modes). Drop PopQA as a separate dataset (1,570 of RetrievalQA's
rows are PopQA questions) and FreshQA from the core (no frozen release, stale answers,
LLM-judge grading). Add Multi-IF and BFCL multi-turn base for multi-turn coverage, and a
long-context probe as its own track.

## 6. Measurement protocol

### 6.1 Job anatomy

The harness lives in `execubench/harness/` and is **scaffolding**: its parts (adb calls with
bounded timeouts, staging with sha256 checks on host and phone, the state sampler, the streaming
client with host monotonic timestamps, crash-safe JSONL records) are unit-tested against stubs,
but the orchestration that runs a job end to end is not written and nothing has run on a phone.
`run.py` refuses to start without a pinned APK hash, refuses an ExecuServe build without
contract 1 or with another APK hash, and otherwise exits with "orchestration not implemented",
never success. The P1 pilot is pinned in `data/pilot/p1.yaml`: four phones (one per Tier A
vendor for the contract tests), two models, six experiments with their workloads, a worst case
of 750 device minutes under a 900-minute ceiling (`tests/test_pilot.py` checks every pin and the
sum). The smoke workload's 20 requests (16 single-turn prompts and two two-turn conversations,
written for this repository) are pinned by sha256 in `data/pilot/smoke-v1.jsonl`; the
determinism experiment repeats each request in the same job and compares units only if Device
Farm happens to assign different ones; the artifact-survival experiment loops until the job's
time limit and states what the pull must hold. Two controls bound spending beyond one run:
`execubench pilot check` lists every blocker (today: no pinned APK, no orchestration, the speed
track's source text not chosen), and each run's worst case is reserved in a ledger
(`execubench/ledger.py`, `--ledger` on scheduling commands) that refuses a reservation past the
pilot's 900 minutes, across experiments and retries.

One Device Farm job is one device unit, one model file, one or more tracks, at most 150
minutes (the service's hard limit). Custom test environment on the Amazon Linux 2 host
(x86_64; all 38 jobs of the first two probes recorded kernel 5.10.269 and adb 1.0.39), test
type Appium Python used only as a carrier for our own test spec, a manifest-only carrier APK
(`devicefarm/carrier/`), and **video capture off** (Device Farm records the screen by default,
and the encoder would compete for CPU and GPU).

1. **Identify.** Run the probe script, record the unit hash, build fingerprint and starting
   temperatures.
2. **Stage.** The host downloads the `.pte` and tokenizer from the Hub at the pinned revision
   (host download of an 11 MB tokenizer measured 12.4 to 35.7 MB/s, median 32.0, across 38
   jobs), checks sha256, pushes into ExecuServe's app-created model folder (a 256 MiB
   zero-filled push measured 25.4 to 89.1 MiB/s, 27 of 38 jobs between 25 and 30, so roughly 15
   to 50 s for a 1.3 GB model, an extrapolation that P1 replaces with real model pushes), and
   checks sha256 again on the phone.
3. **Start.** Install the pinned ExecuServe APK, start the service, `adb forward`, load the
   model, record load time.
4. **Cool down.** Wait until battery temperature is at or below the threshold and thermal
   status is 0, up to a maximum wait (both recorded). In the second probe, before any model
   ran, the HAL's current battery readings were 27.6 to 35.3 C (sensors `BAT`, `battery`,
   `BATTERY`, `MAINBATRAW`; zero-reading `SUBBAT` placeholders excluded), the hottest CPU-type
   sensor per phone 29.5 to 41.3 C, and one Galaxy S24 Ultra was already at thermal status 1.
5. **Warm up.** Three discarded requests.
6. **Tracks**, in a fixed order: speed, then quality or agreement shards, then long-context.
   Each request writes one line to `requests.jsonl` immediately, and the host copies the
   records into `$DEVICEFARM_LOG_DIR` after every request.
7. **Close.** Final state capture, pull everything, unload.

**Artifact limits.** Device Farm keeps customer artifacts only up to 1 GB per job and drops
**all** of them above that ([limits](https://docs.aws.amazon.com/devicefarm/latest/developerguide/limits.html)).
Requests and samples are small (an **estimate** of tens of MB for a full quality shard). The
orchestration, when written, must track the directory size, compress samples, never copy model
files or logcat into it, and record `artifact_bytes`; none of that exists yet. P1's
`artifact_survival` experiment checks what survives a job killed at the time limit; if copying
into the log directory proves insufficient, the host uploads each shard's records to a
presigned S3 URL as it goes. On the pulling side, every archive and every job is bounded in
size, count and member type (`execubench/devicefarm.py`).

### 6.2 State sampling

Every 250 ms during a request, over adb: server process memory (`/proc/<pid>/status`) and
`scaling_cur_freq` per core. Every 1 s: `dumpsys thermalservice` (status and the HAL's current
temperatures). At request start and end: `dumpsys battery`. Per block: `cpufreq`
`time_in_state`. All 19 phones of the second probe expose a thermal status, a skin
temperature and a battery temperature in the HAL's current block, under vendor names (`SKIN`
on Samsung and the Redmi, `skin` on the Xiaomi 13, `VIRTUAL-SKIN` on Pixels). Sampling every
second can still miss a thermal event shorter than a second. Sampling cost on
the phone is measured in P1 by running the speed track with sampling at 250 ms, 1 s and off.

### 6.3 Speed track

Prompt lengths 128, 512, 1,024 and 2,048 tokens (plus 4,096 at 8k windows), exact per model
tokenizer, cut from one public-domain text. Decode targets of 64 and 256 steps (requests ask
for 65 and 257 sampled tokens). A file runs only the buckets where prompt plus decode target
plus one fits its window: in the window sweep, the 2k files run 128, 512 and 1,024 with both
targets (1,024 + 257 fits in 2,048; 2,048 does not), and every larger window runs all buckets
it can hold. KV cache reset per request. Ten measured repetitions per cell
after warm-up, alternating cell order so heat does not line up with one cell. That gives
`n = 10` per cell per job; p95 needs `n >= 20` and p99 needs `n >= 100`, which come from
repeating the track across jobs, units and days, never from extrapolating, and a cell's
interval is a bootstrap over whole jobs (`docs/METRICS.md`).

### 6.4 Quality and agreement tracks

The frozen selection (`suites/v1/`) is sharded into jobs that fit the 150-minute limit with a
20 percent margin. Greedy decoding (temperature 0), thinking off for every family that has a
switch (`enable_thinking: false` for Qwen3), the per-dataset output caps and tool-loop limits
in `docs/DATASETS.md`, eligibility fixed per model before collection. Single-turn requests
reset the KV cache; multi-turn conversations reuse it, as an app would, and record
`cached_tokens`.

- **Reference runs:** the full selection on every Tier A phone. Quality is published **per
  device**.
- **Agreement runs:** the first 40 IDs per dataset and mode on every other phone in the
  standard, reported as observed agreement with the same-vendor Tier A phone and with the x86
  host run. Never extrapolated to rows not run: zero mismatches in 40 still allows a
  disagreement rate up to about 7 percent.
- **Host runs:** the same `.pte` through ExecuTorch's Python `TextLLMRunner` on an x86 machine
  (as execupack's smoke test does), and the source checkpoint in fp32 through Transformers with
  the same template and greedy decoding. The first catches harness bugs before device minutes
  are spent and gives a third agreement point; the second measures deployment fidelity, what
  export and quantization cost in accuracy.

### 6.5 Fixed performance mode

`cmd power set-fixed-performance-mode-enabled true` was accepted (exit 0, no output) on every
probed phone. Accepted means the PowerHAL took the hint, not that clocks hold. P1 runs the speed
track with it on and off, alternating, on three phones; it becomes the default only if it
measurably narrows the spread without lowering the median by more than the spread it removes.
Every job records which way it ran, and it is part of the cell key.

### 6.6 Units, days and spread

Every result records the unit hash and the build fingerprint. A cell is published with the
number of jobs, distinct units and distinct days behind it. Across the first two probes, 17
of 19 models landed on a different physical unit the second time (`unit.txt` in each probe
folder). P1 measures speed spread between units directly: the same speed track on the same
phone model in five jobs.

Prior evidence, from OpenWeights' research note
([`docs/research/executorch-state-and-recipes.md`](https://github.com/ExperimentalMachines/openweights/blob/59e17f24696c49a495ed0281c9d0e348c082a3f9/docs/research/executorch-state-and-recipes.md),
Firebase Test Lab, 2026-09-18; the underlying run files are in that repository, not here): two
Galaxy S25 Ultra units differed by a third on prefill, with battery temperatures of 28.9 and
18.7 C as their sessions began; and the same `.pte` gave column totals that matched on two
units while 14 questions differed in correctness underneath, and a third unit moved
correctness by about four points. Until P1 says otherwise, assume speed spreads of that size
between units, and do not assume greedy outputs are identical across units of one model. P1
therefore also runs a **determinism check**: the same 20 requests twice on one unit, then on a
second unit of the same model.

## 7. Grid and cost

### 7.1 The v1 grid

| Track | Files | Devices | File by device combinations |
|---|---|---|---|
| Speed at 8k | 16 (one per repo) | 11 (the standard) | 176, each with 10 prompt-by-decode buckets |
| Window sweep | 13 (three models, every published window) | 2 (S25 Ultra, Pixel 10) | 26 |
| Quality reference at 8k | 16 | 4 (Tier A) | 64, full selection |
| Agreement at 8k | 16 | 7 (Tiers A+, B and C) | 112, 40 IDs per dataset and mode |
| Long-context | 32 (the 8k file and the largest published window, per repo) | 2 (S25 Ultra, Pixel 10) | 64 |

The Galaxy A17 runs only the models whose files load in its memory; the others are cells marked
"not applicable: does not fit", which is itself a finding once measured. `docs/GRID.md`
(generated in P2) lists every cell with its status: planned, running, measured, failed with
reason, or not applicable with reason.

### 7.2 Cost model

From `python -m execubench budget` (`execubench/budget.py`). **Every input is an estimate**
until P1 measures it: 2,800 generations per model for the full quality selection (provisional
until P2 freezes it), 300 per agreement run, 15 s per generation on a flagship and three times
that on a mid-range phone, 15 minutes per speed track, 20 minutes per long-context file, 8
minutes of overhead per job, 120 usable minutes per job, and 20 percent for retries.

| Track | Device minutes | Jobs |
|---|---:|---:|
| Quality reference | 67,200 | 576 |
| Agreement | 15,600 | 160 |
| Speed | 4,560 | 176 |
| Window sweep | 390 | 26 |
| Long context | 1,280 | 64 |
| Job overhead | 8,016 | |
| **Total before contingency** | **97,046** | **1,002** |
| **With 20 percent contingency** | **116,455** (about 1,941 device hours) | |

Jobs are rounded up per model and device, because a job holds one model file on one unit:
leftover minutes of different models or devices cannot share a job.

- **Metered** at **$0.17 per device minute**
  ([pricing](https://aws.amazon.com/device-farm/pricing/)): about **$19,800**.
- **Unmetered** at **$250 per Android slot per month**, where a slot is one concurrent device of
  any model: 1,941 hours is 2.7 slot-months at full use and 3.9 at 70 percent use, so about
  **$1,000** (for example four slots for one month).
- The full quality selection on Tier A is about 69 percent of the work (67,200 of 97,046
  minutes), and the mid-range A56 alone is half of that, because it is assumed three times
  slower. The levers, if the budget must shrink, are a smaller
  frozen selection or fewer phones with the full selection.
- The free trial was 1,000 minutes. The six probe runs used 54.16 metered minutes (8.41,
  9.24, 8.86, 25.33, 1.11 and 1.21, from each run's `devicefarm-run.json`), and the account
  then reported 945.8 remaining (`data/devicefarm/account-2026-10-04.json`). P1 is an **estimate**
  of 600 to 900 minutes, so it should fit but is not guaranteed to.

**Recommendation:** run P1 metered, then buy unmetered slots for P3 if P1's measured minutes
confirm the estimate.

## 8. Data flow and storage

```
Device Farm job ($DEVICEFARM_LOG_DIR)
  probe/          raw device dumps
  job.json        harness run record (schemas/run.schema.json)
  requests.jsonl  one line per generation (schemas/request.schema.json)
  samples.jsonl   state samples (schemas/sample.schema.json)
  server.log      ExecuServe and runner logs
        |  execubench devicefarm pull --kind harness <run-arn> <new folder>
        |  (staged and renamed atomically; identifiers redacted in the probe dumps and in
        |   Device Farm's records; any identifier elsewhere refuses the pull; a manifest
        |   lists every file's sha256)
        v
restricted raw storage, outside this repository   (docs/DATA-POLICY.md)
        |  execubench validate --runs <pulled run> (schemas + semantic checks with the
        |  raw-record rules; refuses an incomplete pull or a folder with no job), then grading, then the
        |  publication exporter (identifier rescan, per-dataset license rules: Multi-IF
        |  prompts become a hash with replay: "restricted")
        v
data/runs/ (publishable records), data/results/v1/*.json, docs/GRID.md
        |  release
        v
Hugging Face dataset experimentalmachines/execubench-results (publishable records + summaries)
```

Grading, the summariser and the exporter are P2 deliverables. Probe evidence, which has no
dataset content, goes straight to `data/devices/probe/`. Dataset rows whose license forbids
redistribution are never committed; manifests carry IDs and hashes.

## 9. Phases

| Phase | Goal | Exit criterion |
|---|---|---|
| **P0 Probe** (done 2026-10-04) | Know what Device Farm's phones are | Every arm64 Android model in the catalogue probed; findings in `docs/DEVICES.md` |
| **P1 Pilot** | Prove the protocol on 4 phones and 2 models (`data/pilot/p1.yaml`) | `execubench pilot check` reports no blocker before the first job; an ExecuServe build passes every contract-1 acceptance test (`docs/EXECUSERVE-CONTRACT.md`) on a phone of each Tier A vendor; every field in `schemas/` filled from a real run and `execubench validate --runs` clean on it; ExecuServe changes merged; adapter contract tested on the four request shapes; sampling overhead, fixed-performance A/B, unit spread, determinism, `/proc` readability, per-process counters and artifact survival measured; cost model inputs replaced by measured minutes |
| **P2 Freeze** | Freeze suite v1 | Model manifest and `suites/v1/*.ids.json` with hashes; eligibility per model; graders vendored and tested against upstream examples; host runs of all 16 models; Codex review of the frozen suite |
| **P3 Collect** | Fill the v1 grid | `docs/GRID.md` shows no planned cell; failures carry reasons |
| **P4 Publish** | Release | Results dataset, tables, methodology; Codex review of every table against the raw files |

Open owner decisions before P3: unmetered slots or metered; whether the Exynos stand-in
(Galaxy A56) stays in Tier A or Exynos leaves Tier A until Device Farm offers a newer one; and
whether the full quality selection runs on all four Tier A phones (Section 7.2).

## 10. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| No root: clocks and thermal state drift | Speed spread | Sample and publish conditions, cooldown gate, mark thermal events and clock drops, repeat across units and days |
| Device Farm unit heterogeneity, including firmware | One model reads differently by unit | Unit hash and build in every record; device id includes the build; P1 measures spread |
| The Hub republishes exports | A "model" changes under a result | Results name sha256; P2 freezes a model manifest |
| Runner clock is `CLOCK_REALTIME` | A time adjustment corrupts a duration | Monotonic cross-check per request; disagreeing requests excluded from speed |
| Device Farm catalogue changes | Grid cells become unrunnable | Dated catalogue snapshots; re-probe monthly; a changed fingerprint is a new device id |
| ExecuServe changes alter timings | Results not comparable across versions | APK hash in the cell key; timing semantics rechecked on every runtime bump |
| Charging heat on always-plugged phones | Battery temperature biased up | Recorded and stated; skin and SoC temperatures recorded beside it |
| Greedy outputs differ across CPUs or units | Quality not portable across phones | Quality published per device; agreement measured, never extrapolated |
| 150-minute job limit and the 1 GB artifact cap | Shards or all artifacts lost | Shards sized with a 20 percent margin; artifact size tracked; S3 fallback |
| Model templates differ from upstream | Unfair quality | Templates byte-tested in ExecuServe; the rendered prompt kept and hashed per request |
| Runtime upgrades change outputs | 1.4.0 and 1.5.1 numbers mixed | Runtime version in every cell key; host compatibility reports per file (`data/runtime/`) |
| Stock Android API cannot cap output or guard the window | Overlong replies; KV overflow in multi-turn | Contract 1 requires a native cap and a server-side capacity guard, tested before P1 |
| Pulled artifacts contain identifiers | Leak on publication | Pulls refuse any identifier outside the probe dumps; publication exporter rescans (`docs/DATA-POLICY.md`) |

## 11. Open questions

1. Does fixed performance mode steady clocks on any of these phones? (P1)
2. Can the shell read the server's `/proc/<pid>/status` on every phone? (P1)
3. Does user-mode or per-process `simpleperf` counting work on a `profileable` build? (P1;
   probe v3 also records a user-mode system-wide attempt)
4. ~~Does the Java `LlmModule` expose `ignore_eos`?~~ Answered from the sources: neither 1.4.0
   nor 1.5.1 does, and neither forwards `maxNewTokens`. The speed track excludes early stops and
   the contract requires a native cap.
5. How large is unit-to-unit spread on Device Farm compared with Test Lab? (P1)
6. Are greedy outputs byte-identical across runs on one unit, across units of one model, and
   across SoCs, for the same `.pte`? (P1 determinism check, P3 agreement track)
7. Is `time_in_state` readable on every phone? (readable on the phones of the two script
   checks; the next full probe records it everywhere)

## 12. Review log

| Date | Reviewer | Subject | Finding | Disposition |
|---|---|---|---|---|
| 2026-10-04 | Codex gpt-6.1-sol (medium, web search) | Proposed datasets | Keep GSM8K, IFEval, BFCL subset; replace RetrievalQA with SQuAD 2.0; PopQA optional; drop FreshQA; add Multi-IF, long context, fidelity gate | Adopted except RetrievalQA: kept (expanded release) because its frozen contexts and answerability labels give a tool-decision measure SQuAD lacks. Row counts and revisions re-verified by us, then recomputed by `execubench datasets stats` |
| 2026-10-04 | Codex gpt-6.1-sol (medium, web search) | Published specs for 12 SoCs and 19 phones | No maker publishes BF16/FP16 TFLOPS or DRAM bus width for these chips; Tab S11 is the 9400+; Pixel 9 GUR25 is not the US model code; SME absent from X925 | Adopted: those columns stay `unknown`; Qualcomm, MediaTek, Samsung and Google pages spot-checked by us; `checked_by` records who confirmed each value |
| 2026-10-04 | Codex gpt-6.1-sol (medium) | Whole repository, first draft (25 findings) | Stale cached temperatures used as readings; budget contradicted its inputs; scrubber left IMEIs and serials; schemas admitted unpublishable records; device ids ignored firmware; per-request peak RSS was a process high-water mark; cell key too coarse; 40-row agreement over-extrapolated; correlated requests treated as independent; throttling flag claimed a cause; a counter metric was called a frequency; runner clock is `CLOCK_REALTIME`; fp32 reference called contamination-immune; caps and eligibility underspecified; RetrievalQA truncation could remove the evidence; tool-decision metrics conflated; dataset numbers not reproducible locally; pulls could overwrite evidence and trusted archive paths; probe PASSED did not mean collection succeeded; unknown core designs counted as big; mixed provenance; catalogue-wide Exynos claim unsupported; grid expansions missing; artifact size limit ignored | Revised in commit `ad4847a`. Codex's second round judged 11 fixed and 14 partly fixed (next row) |
| 2026-10-04 | Codex gpt-6.1-sol (medium) | Second round on `ad4847a` (14 findings) | Identifier redaction still missed MACs, other serial keys, unique numbers, fingerprint UID and camera fuse IDs; pull wrote the run record before refusing a used folder and let archives collide; successful requests validated with empty runner stats, null memory and no prompt; budget rounded jobs across models; speed workload and effective threads optional; a total-duration clock check cannot protect stage timings; long-context scope contradicted DATASETS and 2k sweep files could not hold the prompts; `answer_in_context` had no schema field; registry dropped older firmware builds; an all-A53 phone was labelled big.LITTLE; probe coverage overstated; the 68-file scan, FreshQA and MT-Eval counts and trial minutes had no local evidence; README still said a column never mixes provenance; the round-1 disposition overstated completion | Revised in `211a173`. Codex's third round judged 8 fixed (pulls, budget rounding, speed workload and threads, stage-clock specification, long-context scope, retained builds, architecture labels, local evidence) and 6 partly fixed (next row). Deriving the Hub change by script corrected one number: three repos, not four, moved to GPTQ |
| 2026-10-04 | Codex gpt-6.1-sol (medium) | Third round on `211a173` (7 findings) | `ro.boot.cpuid` and a multi-line `ro.boot.chipid` survived redaction; BFCL step-limit wording differed from upstream's counting; thermal status could be null without a reason; RetrievalQA retention fields could be null and the subset split empty; probe coverage compared counts, not sets, and staging failures were unclassified; one BF16 explanation stayed universal (an earlier replacement had silently not applied); the review log said "all addressed" | Addressed in `8159cc5`; Codex's fourth round judged 5 fixed and 2 partly fixed (search-tool retention rules were prose only; this cell overstated closure). Both corrected in the commit after it: the schema now ties search-tool retention fields to an explicit `tool_called`, with tests. What `8159cc5` did: `cpuid`, `soc_id` and `board_id` keys, multi-line property parsing in both the scrubber and the parser, tests on the real formats plus an identifier scan independent of the key list, all five runs re-pulled; DATASETS now states upstream's exact step counting, with a P1 boundary test; thermal status needs a value or a reason; with-context requests need a boolean and at least one passage hash, a published split needs all four counts; probe v3.2 compares exact CPU sets with `cpu_present` (dry-run on four real dumps and one negative case, then run on Device Farm on a Pixel 11 and a Pixel 2 XL, 2 of 2 passed) and labels the Hub fetch optional; every edit script now asserts its target. Still open by design: everything marked P1 in Sections 6 and 11, and execuserve changes in Section 3.3 |
| 2026-10-04 | Codex gpt-6.1-sol (medium, web search) | Full QA of `6b54928` and the ExecuTorch 1.5.x upgrade (26 findings) | Scrubbing rewrote every artifact, which would falsify content hashes; Android 1.5.1 does not forward `maxNewTokens`; `seqLen` is unsafe on these exports (sliding-window branch); `validate` ignored real records; schemas admitted impossible values; JNI UTF-8 buffering breaks first-callback timing; Device Farm metadata bypassed redaction; 1.4.0-to-1.5.1 compatibility unproven; no harness or contract; vendored GPL file without its license; incomplete pulls reported as success; no partial-pull recovery; no deadlines or retries; unbounded archives; unhashed dependencies; predictable temp paths; inventory test trusted a flag; a failed scan overwrote the inventory; no pilot pool, manifest or spend guard; no timing coordinates; optional publication fields; publication policy prose only; regeneration claims overstated; packaging; parser edge cases; repository controls | Verified the three runtime findings in the v1.5.1 sources ourselves, then addressed all 26 in `353e15a` (contract document, fail-closed pulls, semantic validator, host compatibility runs, locks, pilot manifest and the rest). Codex's verification round judged 9 fixed and 17 partly fixed, and found 17 further problems (next row) |
| 2026-10-04 | Codex gpt-6.1-sol (medium, web search) | Verification of `353e15a` (17 findings) | A bundle without probe identity disabled the leak check; JSON-escaped identifiers survived metadata redaction; the harness returned success without running; impossible records still validated (negative percentiles, accuracy with no attempts, ok with an error, out-of-order host times, sampled over the cap); `finalized` could be omitted; retried `schedule_run` could start two runs; pull completeness ignored the job count and expected files, and failed pulls left no manifest; samples were associated by a mutable sequence number and lost raw readings; the client accepted cut and error streams and did not assemble tool calls; compatibility coverage and repeatability claims exceeded the evidence; archive bounds were per archive; socket timeouts were not deadlines and URL refresh relied on order; sampler shutdown raced the writer; the memory mean ignored edges; data policy and plan disagreed on the data flow; probe timing and signals; empty or conflicting inventory scans | All 17 addressed in the next commit, each with tests: identity required before any harness file is accepted; recursive sanitising plus an escaped-form scan; `run.py` exits 4 "orchestration not implemented"; new semantic and schema rules; complete jobs must be finalized and hashes are recomputed; `schedule_run` without retries plus a lookup by name after an ambiguous failure; job-count and expected-file reconciliation with a failure manifest; samples carry read intervals and raw thermal lines and are matched by host time; strict stream completion and tool-call assembly; the compatibility summary lists the full v1 grid and a repeat check backs the repeatability sentence; job-level bounds and non-regular member refusal; total download deadlines and refresh by artifact ARN; sampler stop waits and surfaces failures; step-function memory mean with coverage; one data flow in PLAN and DATA-POLICY; probe timing validation and terminating signal traps; empty, duplicate and conflicting scans refused. Explicitly not done: verified-entry resumption of a failed pull (a failed pull is moved aside and repeated instead), the harness orchestration, the exporter, grading and the summariser (P1 and P2) |
