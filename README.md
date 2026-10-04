# execubench

**Benchmarks for compiled ExecuTorch models on real phones.** Speed, memory, heat and answer
quality, single turn and multi turn, measured on AWS Device Farm, with every number traceable
to a raw file from a device run.

> [!NOTE]
> Status: **planning and probing (phase P0 done).** No benchmark results yet. The plan, the
> metric definitions, the dataset choices and the device standard are written down and
> reviewed, and every arm64 Android model in Device Farm's catalogue has been probed. The
> pilot (P1) is next. See [`docs/PLAN.md`](docs/PLAN.md).

An independent project by [Experimental Machines](https://experimentalmachines.org). Not
affiliated with, endorsed by or sponsored by the PyTorch Foundation, Meta or Amazon.

## What it measures

| Per phone | Per request | Per result cell |
|---|---|---|
| SoC, CPU cores, designs (from MIDR) and clocks, clock-domain topology, all-big-core or not | TTFT, prefill and decode throughput (from ExecuTorch's runner timestamps) | p25, p50, p75, p95, p99 with `n` |
| GPU (Vulkan device name), NPU and the vendor NPU runtimes present | Peak and mean memory (RSS, anon versus file, PSS) | Accuracy with a Wilson 95% interval |
| Usable and marketed RAM, memory type where stated | Battery, skin and SoC temperatures, thermal status, sampled CPU clocks | Tool-call validity, correctness and decision accuracy |
| Memory bandwidth and BF16 TFLOPS: only where a maker publishes them (none does, so far) | Raw output, parsed tool calls, finish reason | Device units and jobs behind the cell |

Every value carries its own provenance: `measured`, `reported` (by the phone), `published`
(by a chip or phone maker, with a URL), `derived` (with its formula) or `unknown`. A published
table shows the label per cell or splits the column by label. Definitions:
[`docs/METRICS.md`](docs/METRICS.md).

## What it runs

- **Models:** every XNNPACK `.pte` exported by
  [execupack](https://github.com/ExperimentalMachines/execupack) to the
  [`experimentalmachines`](https://huggingface.co/experimentalmachines) Hugging Face org:
  79 files in 16 repos at the last scan (SmolLM2, Qwen2.5, Qwen3, Llama 3.2, LFM2.5), pinned
  by revision and sha256 in [`data/models/xnnpack.json`](data/models/xnnpack.json). The Hub
  changes as execupack republishes, so results always name a file's sha256.
- **Runtime:** ExecuTorch **1.5.1** (`config/versions.env`), through
  [ExecuServe](https://github.com/ExperimentalMachines/execuserve) on the phone
  (OpenAI-compatible API), driven by a harness on the Device Farm host over `adb forward`, so
  multi-turn tool loops can execute real tool simulators between turns. ExecuServe must first
  meet the benchmark contract in [`docs/EXECUSERVE-CONTRACT.md`](docs/EXECUSERVE-CONTRACT.md):
  the stock 1.5.1 Android API neither caps output nor guards the context window on these
  exports. The files are 1.4.0 exports; their greedy output changes under the 1.5.1 runtime
  (`data/runtime/`), so results are always tied to a runtime version.
- **Datasets:** GSM8K, IFEval, Multi-IF, BFCL (v4 categories, including multi-turn) and
  RetrievalQA in closed-book, with-context and search-tool modes, plus a long-context probe
  and a controlled speed track. Why these, and why PopQA and FreshQA were folded in or
  dropped: [`docs/DATASETS.md`](docs/DATASETS.md).
- **Phones:** a four-vendor Tier A (Galaxy S25 Ultra, Pixel 10, Galaxy Tab S11, Galaxy A56),
  current-generation Tier A+ (Galaxy S26 Ultra, Pixel 11), mid-range Tier B and earlier
  anchors in Tier C. What each phone reports about itself, and why Device Farm's own
  catalogue is not trusted: [`docs/DEVICES.md`](docs/DEVICES.md).

## Repository layout

| Path | What |
|---|---|
| `docs/PLAN.md` | Decisions, protocol, grid, cost model, phases, risks, review log |
| `docs/METRICS.md`, `docs/DATASETS.md`, `docs/DEVICES.md` | Definitions, dataset verdicts, device standard |
| `schemas/` | JSON Schemas for phone, job, request and summary records, with examples |
| `devicefarm/probe/` | The device probe: a shell script run on the Device Farm host against the phone over adb |
| `devicefarm/carrier/` | A manifest-only carrier APK (minSdk 21) for Device Farm's app-based test types |
| `execubench/` | Python package: Device Farm plumbing (fail-closed pulls, spend guard and ledger, pilot gate), probe parser, device registry, model inventory, schema and semantic validation, host runtime runs |
| `execubench/harness/` | The benchmark harness for the Device Farm host (unit-tested, not yet run on a phone) |
| `docs/EXECUSERVE-CONTRACT.md`, `docs/DATA-POLICY.md` | What the phone server must provide; what is published and kept |
| `data/pilot/p1.yaml` | The pinned P1 pilot: devices, models, experiments, spend ceiling |
| `data/runtime/` | Host runs of the same files under two ExecuTorch versions, compared piece by piece |
| `config/versions.env`, `requirements/*.lock` | The runtime pin; hash-pinned dependency sets |
| `LICENSES/`, `THIRD_PARTY_NOTICES.md` | Licenses of third-party material (one GPL-2.0-only data file) |
| `data/devices/` | Raw probe dumps, catalogue snapshot, published specs with sources (`specs.yaml`), the standard (`standard.yaml`), generated `devices.json` |
| `data/models/xnnpack.json` | Generated model inventory |
| `data/datasets/stats.json` | Every dataset figure the docs cite, recomputed from pinned files with hashes |
| `data/reference/` | Pinned third-party references (Linux `cputype.h` for MIDR names) |

## Setup

The tool runs from a checkout (it reads `schemas/`, `data/` and `docs/` beside the package).
One environment, with the hash-pinned dependencies CI uses:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements/dev.lock
.venv/bin/python -m execubench validate
```

Host runs of `.pte` files need a separate environment from `requirements/host.lock`
(ExecuTorch 1.5.1 and its torch), and dataset recounts add `requirements/datasets.lock`.
`uv run --with-requirements requirements/dev.txt <command>` also works for one-off commands.

## Commands

```sh
uv run --with-requirements requirements/dev.txt pytest -q          # tests (no network)
uv run --with-requirements requirements/dev.txt python -m execubench validate   # schemas + semantic checks
python -m execubench host run <pte> <tokenizer> --out r.json       # in a venv from requirements/host.lock
python -m execubench models scan                                   # re-pin the Hub inventory
python -m execubench datasets stats                                # recount dataset figures (needs requirements/datasets.txt)
python -m execubench budget                                        # the v1 device-minute budget
python -m execubench pilot check data/pilot/p1.yaml                # what still blocks the P1 pilot
python -m execubench devicefarm probe <project-arn> <pool-arn> --carrier-apk carrier.apk --max-device-minutes 30
python -m execubench devicefarm pull <run-arn> data/devices/probe/<new folder>   # exit 2 if incomplete
python -m execubench validate --runs <pulled harness run>          # restricted records, before grading
python -m execubench devices build data/devices/probe/<run>...     # oldest first
python -m execubench devices table                                 # regenerate the table in docs/DEVICES.md
```

Device Farm runs in `us-west-2` only. Hub and Device Farm commands need credentials in the
environment (the Hugging Face token from `huggingface_hub`'s own store or `HF_TOKEN`, AWS from
the usual chain); nothing in this repo stores either.

## License

Apache-2.0 for execubench's own code and documentation. One vendored file,
`data/reference/linux-cputype.h`, is GPL-2.0-only and is distributed under that license
(`LICENSES/GPL-2.0-only.txt`); see `THIRD_PARTY_NOTICES.md`. Dataset rows are never
redistributed here; manifests carry IDs and hashes, and each dataset's own license applies
(Multi-IF's data is CC-BY-NC-2.0).
