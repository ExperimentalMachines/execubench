# Third-party material

execubench's own code and documentation are Apache-2.0 (`LICENSE`). The files below are
third-party and keep their own licenses.

| Path | Origin | License | How it is used |
|---|---|---|---|
| `data/reference/linux-cputype.h` | Linux kernel, `arch/arm64/include/asm/cputype.h` at commit `a74306e2e676f9775457366fc047a660fbf02f26` (`data/reference/linux-cputype.h.source`). Copyright (C) 2012 ARM Ltd. | **GPL-2.0-only**; full text in `LICENSES/GPL-2.0-only.txt` | Copied verbatim and read as data: `execubench/devices.py` parses its `#define` lines to name CPU designs. It is not compiled or linked into anything. Distributing this repository distributes the file under GPL-2.0-only; the license of the rest of the repository is unchanged |
| `data/devices/probe/**` | Output of commands run on AWS Device Farm phones (getprop, /proc, sysfs, dumpsys) | Facts reported by the devices; vendor strings remain their owners' | Raw evidence, with identifiers redacted at pull time |
| `data/devices/catalogue/*.json` | AWS Device Farm `list-devices` responses | AWS service output | Snapshot of what the service offered on a date |
| `data/runtime/**` | Greedy outputs of the models named in each file, on license-free prompts written for this repository | The models' own licenses apply to their outputs where relevant (see each source model card) | Evidence that two ExecuTorch runtimes agree or disagree |

No dataset rows are committed. `data/datasets/stats.json` holds counts and sha256 hashes only;
each dataset's license (`docs/DATASETS.md`) applies to the data itself, and Multi-IF's data is
CC-BY-NC-2.0.
