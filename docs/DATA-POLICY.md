# Data policy: what is collected, where it lives, what is published, how long it is kept

## Classes

| Class | What | Where | Published |
|---|---|---|---|
| **Restricted raw** | Full harness records as pulled: rendered prompts, model outputs, tool traces, samples | Maintainer storage outside this repository (a private bucket), keyed by run id | Never as is |
| **Probe evidence** | The probe's device dumps, identifiers redacted at pull time | `data/devices/probe/` | Yes |
| **Publishable records** | Harness records passed through the publication exporter (below) | `data/runs/` (small) or the results dataset | Yes |
| **Summaries** | Cells computed from publishable records | `data/results/` and the results dataset | Yes |
| **Snapshots** | Device Farm catalogue, account settings (account number masked), Hub inventory, dataset counts | `data/` | Yes |

## Checking restricted records

`execubench validate --runs <pulled run>` checks a pull in restricted storage before anything is
graded: the pull manifest must say complete and harness, every job folder is checked against the
schemas and the semantic rules, and raw records must carry their prompt text (only the exporter
may remove it). A folder with no job in it is an error, not an empty success.

## The publication exporter

Records leave restricted storage only through the exporter (P2 deliverable), which:

1. Re-runs the identifier scan of `execubench/devicefarm.py` on every field and refuses the
   record on any hit.
2. Applies each dataset's license rule to prompt and output text:
   - GSM8K (MIT), IFEval (Apache-2.0), BFCL (Apache-2.0), RetrievalQA (MIT declared): prompt
     text may be published.
   - Multi-IF (data CC-BY-NC-2.0): prompt text is replaced by the row id and the prompt's sha256;
     model outputs are published.
   - Anything with an unverified license: prompt text removed, hash kept.
3. When prompt text is removed, sets a `replay: "restricted"` marker so that a reader knows the
   hash cannot be checked against published bytes; the schema then forbids `prompt_text`, and
   the raw-record check forbids the marker before export.
4. Writes an export manifest listing every record, its source run and the rules applied.

## Retention

| Data | Kept for | Then |
|---|---|---|
| Device Farm uploads (APKs, test packages, specs) | Until the run they served is pulled and verified | Deleted with `aws devicefarm delete-upload` |
| Device Farm runs and their artifacts | 30 days after a verified pull (`pull-manifest.json` complete) | Deleted with `aws devicefarm delete-run`; the pulled copy is the record |
| Restricted raw records | Until v1 results are published plus one year, for regrading | Deleted, keeping the export manifest |
| Failed pull staging (`*.partial`, `*.failed-*`) | Until the pull is repeated successfully | Deleted by the maintainer after inspection |
| Published data | Indefinitely | Corrections are new versions; old versions stay addressable |

No data about people is collected: Device Farm phones are AWS-owned test devices, and the
prompts come from public datasets or this repository.
