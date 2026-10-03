# Working in this repository

execubench benchmarks compiled ExecuTorch models on real phones in AWS Device Farm. It aims to
be cited, so the rules below are about not publishing a wrong number. Read `docs/PLAN.md`
first; it holds the decisions and the phase status. Update it when a decision or a
measurement changes.

## Commands

```sh
uv run --with-requirements requirements/dev.txt ruff check . && uv run --with-requirements requirements/dev.txt ruff format --check .
uv run --with-requirements requirements/dev.txt pytest -q
uv run --with-requirements requirements/dev.txt python -m execubench validate
```

CI runs the same three. Device Farm (`us-west-2`) and Hugging Face commands run only from a
maintainer's machine, never in CI.

## Rules that are not visible in the code

- **Provenance.** Every value is `measured`, `reported`, `published`, `derived` or `unknown`
  (`docs/METRICS.md`). Never fill a field from a different kind of source, and never fill an
  `unknown` with a guess or a secondary source. Device Farm's `list-devices` catalogue is not
  a spec source: its `memory` is storage and its `cpu.clock` is one cluster's clock.
- **Raw first.** Phones and the Device Farm host only collect. Parsing, grading and
  summarising happen afterwards from the pulled artifacts. Never edit files under
  `data/devices/probe/` or `data/runs/`; regenerate derived files (`profile.json`,
  `devices.json`, `xnnpack.json`) with the CLI. Tests fail when a derived file is stale.
- **Grid before results.** Before reporting any result, write the full grid (model by window
  by device by track) and mark every empty cell. Partial coverage is never a finding.
- **Check claims against the files.** Every sentence in `docs/` that states a number names
  the file it came from, or is labelled an estimate. Re-derive it before committing. An
  independent Codex review (`codex exec -m gpt-6.1-sol -c model_reasoning_effort='"medium"'
  ... < /dev/null`) runs on every document and table before it is published; log the
  dispositions in `docs/PLAN.md` Section 12.
- **Secrets and identifiers.** No tokens in files, commits or logs. Pulled artifacts have unit
  serials replaced by a hash and the AWS account number masked (`execubench/devicefarm.py`);
  keep it that way.
- **Dataset licenses.** Do not commit dataset rows. Manifests carry IDs and hashes.
- **Prose style.** No em dashes or en dashes anywhere (a test enforces it). Recast the
  sentence with a comma, a colon or a full stop.
- **Device minutes cost money.** $0.17 per device minute after the free trial. Probe on a
  small pool first; never schedule a full pool to test a script change.

## Git

Commit subjects: a scope, a colon and what changed and why
(`probe: read the GPU through Vulkan, because Pixel 11 prints no GLES renderer`). The body
says what was measured or found. Agents append their own attribution trailer.
