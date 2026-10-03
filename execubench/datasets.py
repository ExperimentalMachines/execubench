"""Recompute every dataset figure that docs/DATASETS.md cites, from pinned files, with hashes.

`python -m execubench datasets stats` downloads each pinned file, hashes it, and writes the
counts to data/datasets/stats.json. Dataset rows are never written to the repo; only counts,
hashes and the revisions they came from.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import urllib.request
from pathlib import Path

HF = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl"),
    "multi_if": ("facebook/Multi-IF", "0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa", "multiIF_20241018.csv"),
    "retrievalqa": ("aialt/RetrievalQA", "a6ad065f05fccdecc1f2974ff471929a39966d2f", "retrievalqa.jsonl"),
    "retrievalqa_original": ("zihanz/RetrievalQA", "fe71a76cdc8b46fcb795e8abac64408753d1a0d3", "retrievalqa.jsonl"),
    "popqa": ("akariasai/PopQA", "098765c79ea10a2cb19c828324e33281b8336ec0", "test.tsv"),
}
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_FILES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance", "multi_turn_base"]
BFCL_RAW = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/berkeley-function-call-leaderboard/"
    "bfcl_eval/data/BFCL_v4_{name}.json"
)


def _hf(key: str) -> tuple[bytes, dict]:
    from huggingface_hub import hf_hub_download

    repo, rev, name = HF[key]
    blob = Path(hf_hub_download(repo, name, repo_type="dataset", revision=rev)).read_bytes()
    return blob, {"repo": repo, "revision": rev, "file": name, "sha256": hashlib.sha256(blob).hexdigest()}


def _passage_chars(passage) -> int:
    # Passages are {"title", "text"} objects except in some rows, where they are bare strings.
    if isinstance(passage, str):
        return len(passage)
    return len(passage.get("title", "")) + len(passage.get("text", ""))


def _pct(values: list[int], q: float) -> float:
    """Type 7 (linear) percentile, the method docs/METRICS.md fixes for every percentile."""
    xs = sorted(values)
    h = (len(xs) - 1) * q
    lo = int(h)
    return xs[lo] + (h - lo) * (xs[min(lo + 1, len(xs) - 1)] - xs[lo])


def stats() -> dict:
    import pyarrow.parquet as pq

    out: dict = {}

    blob, src = _hf("gsm8k")
    out["gsm8k"] = {**src, "rows": pq.read_table(io.BytesIO(blob)).num_rows}

    blob, src = _hf("ifeval")
    out["ifeval"] = {**src, "rows": sum(1 for line in blob.splitlines() if line.strip())}

    blob, src = _hf("multi_if")
    rows = list(csv.DictReader(io.StringIO(blob.decode())))
    out["multi_if"] = {
        **src,
        "rows": len(rows),
        "english_rows": sum(1 for r in rows if r["language"] == "English"),
    }

    blob, src = _hf("retrievalqa")
    rqa = [json.loads(line) for line in blob.splitlines() if line.strip()]
    chars = [sum(_passage_chars(p) for p in r["context"]) for r in rqa]
    composition: dict[str, int] = {}
    for r in rqa:
        composition[r["data_source"]] = composition.get(r["data_source"], 0) + 1
    labels: dict[str, int] = {}
    for r in rqa:
        labels[str(r["param_knowledge_answerable"])] = labels.get(str(r["param_knowledge_answerable"]), 0) + 1
    out["retrievalqa"] = {
        **src,
        "rows": len(rqa),
        "param_knowledge_answerable": labels,
        "composition": dict(sorted(composition.items(), key=lambda kv: -kv[1])),
        "context_chars": {
            "definition": "sum over passages of len(title) + len(text); a bare-string passage counts len(string)",
            "method": "linear",
            "p50": statistics.median(chars),
            "p95": _pct(chars, 0.95),
            "max": max(chars),
        },
    }

    blob, src = _hf("retrievalqa_original")
    out["retrievalqa_original"] = {**src, "rows": sum(1 for line in blob.splitlines() if line.strip())}

    blob, src = _hf("popqa")
    popqa = list(csv.DictReader(io.StringIO(blob.decode()), delimiter="\t"))
    questions = {r["question"] for r in popqa}
    pq_rows = [r for r in rqa if r["data_source"] == "popqa"]
    out["popqa"] = {
        **src,
        "rows": len(popqa),
        "retrievalqa_popqa_rows_joined_by_question_text": sum(1 for r in pq_rows if r["question"] in questions),
        "retrievalqa_popqa_rows": len(pq_rows),
        "popqa_question_texts_unique": len(questions),
    }

    bfcl = {"commit": BFCL_COMMIT, "categories": {}}
    for name in BFCL_FILES:
        url = BFCL_RAW.format(commit=BFCL_COMMIT, name=name)
        blob = urllib.request.urlopen(url).read()
        bfcl["categories"][name] = {
            "url": url,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "rows": sum(1 for line in blob.splitlines() if line.strip()),
        }
    out["bfcl_v4"] = bfcl
    return out
