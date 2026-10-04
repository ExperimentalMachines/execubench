# Contributing

Read `AGENTS.md` first: it states the rules that are not visible in the code (provenance, raw
first, grid before results, no em or en dashes). The checks CI runs:

```sh
uv run --with-requirements requirements/dev.txt ruff check .
uv run --with-requirements requirements/dev.txt ruff format --check .
uv run --with-requirements requirements/dev.txt pytest -q
uv run --with-requirements requirements/dev.txt python -m execubench validate
shellcheck -S warning devicefarm/probe/probe.sh devicefarm/carrier/build.sh
```

## Regenerating derived files

Never edit these by hand; regenerate them and commit the result with the change that caused it.

| File | Command |
|---|---|
| `data/models/xnnpack.json` | `python -m execubench models scan` (a scan with problems is written beside it, never over it) |
| `data/datasets/stats.json` | `python -m execubench datasets stats` (needs `requirements/datasets.txt` and network) |
| `data/devices/probe/*/*/profile.json`, `data/devices/devices.json` | `python -m execubench devices build <runs, oldest first>` |
| The table in `docs/DEVICES.md` | `python -m execubench devices table` |
| `docs/PLAN.md` section 7.2 | `python -m execubench budget`, then update the table (a test compares them) |
| `requirements/*.lock` | `uv pip compile requirements/<name>.txt --universal --generate-hashes --python-version 3.11 -o requirements/<name>.lock` |

## Device runs

Device minutes cost money. Every scheduling command requires `--max-device-minutes` and
refuses a run whose worst case (devices x job timeout) exceeds it. Probe on two devices before
a full pool. Pull each run into a new folder; a pull never overwrites evidence.

## Review

Every document change and every published table gets an independent Codex review
(`gpt-6.1-sol`, medium) before merge; record the findings and their dispositions in
`docs/PLAN.md` section 12.
