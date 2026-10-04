"""Run a `.pte` on this machine with ExecuTorch's own C++ `TextLLMRunner` (through its Python
bindings), greedy, and record every piece and the runner's stats.

This is the host side of the agreement track (docs/PLAN.md 6.4) and the runtime-upgrade check:
the same file and prompts under two ExecuTorch versions must give the same tokens before the
phone's runtime moves. Run it in a venv that has exactly one `executorch` version installed;
`python -m execubench host run` writes a JSON report and `host compare` diffs two of them.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

# Fixed, license-free prompts: the point is identical bytes across runtimes, not quality.
PROMPTS = (
    "The capital of France is",
    "Write one sentence about the sea.",
    "1, 2, 3, 4,",
)
MAX_NEW_TOKENS = 48


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(
    pte: Path,
    tokenizer: Path,
    prompts: tuple[str, ...] = PROMPTS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    source: dict | None = None,
    repeat: int = 1,
) -> dict:
    """One fresh runner per prompt and repetition, so no KV or recurrent state carries over.

    `pieces` and `stats` are the first run's. With `repeat` > 1 each prompt runs that many times
    and `repetitions` keeps every run's pieces and stats, so agreement is computed from the
    evidence (`repeat_agreement`), never recorded as a bare flag. With `repeat` 1 the report is
    unchanged from earlier versions."""
    import importlib.metadata

    # The custom and quantized kernels register themselves on import; the runner needs them.
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    from executorch.kernels import quantized  # noqa: F401

    results = []
    for prompt in prompts:
        runs = []
        for _ in range(repeat):
            runner = TextLLMRunner(str(pte), str(tokenizer))
            pieces: list[str] = []
            stats: dict = {}
            config = GenerationConfig(echo=False, max_new_tokens=max_new_tokens, temperature=0.0, num_bos=0, num_eos=0)
            runner.generate(
                prompt,
                config,
                token_callback=pieces.append,
                stats_callback=lambda s, out=stats: out.update(json.loads(s.to_json_string())),
            )
            runs.append((pieces, stats))
        row = {"prompt": prompt, "pieces": runs[0][0], "stats": runs[0][1]}
        if repeat > 1:
            row["repetitions"] = [{"pieces": p, "stats": s} for p, s in runs]
        results.append(row)
    return {
        "executorch": importlib.metadata.version("executorch"),
        "torch": importlib.metadata.version("torch"),
        "source": source or {},
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "pte": {"name": pte.name, "sha256": _sha256(pte)},
        "tokenizer": {"name": tokenizer.name, "sha256": _sha256(tokenizer)},
        "max_new_tokens": max_new_tokens,
        "results": results,
    }


def repeat_agreement(report: dict) -> list[bool]:
    """Per prompt, whether every retained repetition gave the same pieces as the first. A report
    without at least two repetitions per prompt is not repeat evidence and raises."""
    if not report.get("results"):
        raise ValueError("a report without results is not repeat evidence")
    out = []
    for row in report["results"]:
        reps = row.get("repetitions") or []
        if len(reps) < 2 or reps[0]["pieces"] != row["pieces"]:
            raise ValueError(f"prompt {row.get('prompt')!r} has no retained repetitions")
        out.append(all(r["pieces"] == reps[0]["pieces"] for r in reps))
    return out


def compare(a: dict, b: dict) -> dict:
    """Piece-by-piece agreement of two reports made from the same file and prompts."""
    if a["pte"]["sha256"] != b["pte"]["sha256"] or a["tokenizer"]["sha256"] != b["tokenizer"]["sha256"]:
        raise ValueError("reports are for different files")
    if not a["results"] or not b["results"]:
        raise ValueError("a report has no results")
    if a["max_new_tokens"] != b["max_new_tokens"]:
        raise ValueError("reports used different output caps")
    if [r["prompt"] for r in a["results"]] != [r["prompt"] for r in b["results"]]:
        raise ValueError("reports used different prompts")
    rows = []
    for ra, rb in zip(a["results"], b["results"], strict=True):
        if ra["prompt"] != rb["prompt"]:
            raise ValueError("reports used different prompts")
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(ra["pieces"], rb["pieces"], strict=False)) if x != y),
            None if len(ra["pieces"]) == len(rb["pieces"]) else min(len(ra["pieces"]), len(rb["pieces"])),
        )
        rows.append(
            {
                "prompt": ra["prompt"],
                "identical": first_diff is None,
                "first_difference_at_piece": first_diff,
                "pieces": [len(ra["pieces"]), len(rb["pieces"])],
            }
        )
    return {
        "versions": [a["executorch"], b["executorch"]],
        "pte": a["pte"],
        "identical": all(r["identical"] for r in rows),
        "prompts": rows,
    }


def summary_markdown(folder: Path, expected: list[str] | None = None) -> str:
    """SUMMARY.md for a compatibility folder: one row per expected file (the v1 grid), marked
    "not run" when its comparison is missing, then any other file compared there."""
    compared = {}
    for path in sorted(folder.glob("*.compare.json")):
        c = json.loads(path.read_text())
        compared[c["pte"]["name"]] = c
    names = list(expected or []) + sorted(n for n in compared if n not in (expected or []))
    rows = []
    for name in names:
        c = compared.get(name)
        if c is None:
            rows.append(f"| `{name}` | not run | | |")
            continue
        diffs = [p["first_difference_at_piece"] for p in c["prompts"]]
        rows.append(
            f"| `{name}` | {sum(p['identical'] for p in c['prompts'])} of {len(c['prompts'])} | "
            + ", ".join("same" if d is None else f"piece {d}" for d in diffs)
            + f" | {'yes' if c['identical'] else 'no'} |"
        )
    grid = [n for n in (expected or [])]
    done = [n for n in grid if n in compared]
    identical = sum(compared[n]["identical"] for n in done)
    extra = [n for n in compared if n not in grid]
    return (
        f"# {folder.name}\n\n"
        "Generated by `python -m execubench host summary`; do not edit. Each file was run with\n"
        "ExecuTorch's TextLLMRunner, greedy, 48 new tokens, on three fixed prompts, under each\n"
        "runtime in its own venv on the same host; the reports beside this file hold every piece.\n\n"
        f"**v1 grid (every 8k export): {len(done)} of {len(grid)} compared; {identical} of those gave\n"
        f"identical output under both runtimes.** Other files compared: {len(extra)}.\n\n"
        "| File | Prompts identical | First difference per prompt | Identical |\n"
        "|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def v1_files(inventory: Path) -> list[str]:
    """The v1 grid's file names: every 8k export in the inventory (docs/PLAN.md 3.2)."""
    files = json.loads(inventory.read_text())["files"]
    return sorted(Path(f["file"]).name for f in files if f["window"] == 8192)
