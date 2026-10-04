"""config/versions.env is the one place the runtime version lives; everything else must agree."""

import re
import tomllib
from pathlib import Path

from execubench import versions

ROOT = Path(__file__).resolve().parent.parent
ENV = versions.load()


def _pins(path: Path) -> dict[str, str]:
    return dict(re.findall(r"^([A-Za-z0-9_.\-\[\]]+)==([^\s;\\]+)", path.read_text(), flags=re.M))


def test_host_requirements_match():
    for path in (ROOT / "requirements" / "host.txt", ROOT / "requirements" / "host.lock"):
        pins = _pins(path)
        assert pins["executorch"] == ENV["EXECUTORCH_VERSION"], path
        assert pins["torch"] == ENV["TORCH_VERSION"], path


def test_pyproject_extras_match():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    host = dict(p.split("==") for p in project["optional-dependencies"]["host"])
    assert host == {"executorch": ENV["EXECUTORCH_VERSION"], "torch": ENV["TORCH_VERSION"]}


def test_locks_cover_their_direct_pins():
    for name in ("dev", "datasets", "host"):
        direct = _pins(ROOT / "requirements" / f"{name}.txt")
        locked = {_norm(k): v for k, v in _pins(ROOT / "requirements" / f"{name}.lock").items()}
        for package, version in direct.items():
            assert locked.get(_norm(package)) == version, (name, package)


def _norm(name: str) -> str:
    """PEP 503 normalisation, without extras: huggingface_hub and huggingface-hub are one package."""
    return re.sub(r"[-_.]+", "-", name.split("[")[0]).lower()


def test_docs_name_the_runtime():
    for doc in ("docs/METRICS.md", "docs/PLAN.md", "README.md"):
        assert ENV["EXECUTORCH_VERSION"] in (ROOT / doc).read_text(), doc


def test_runtime_compat_summary_is_current():
    from execubench import host

    folder = ROOT / "data" / "runtime" / "compat-1.4.0-vs-1.5.1"
    expected = host.v1_files(ROOT / "data" / "models" / "xnnpack.json")
    assert len(expected) == 16
    assert (folder / "SUMMARY.md").read_text() == host.summary_markdown(folder, expected)
    for compare in folder.glob("*.compare.json"):
        stem = compare.name.removesuffix(".compare.json")
        a = (folder / f"{stem}.executorch-1.4.0.json").read_text()
        b = (folder / f"{stem}.executorch-1.5.1.json").read_text()
        import json

        assert json.loads(compare.read_text()) == host.compare(json.loads(a), json.loads(b))


def test_compare_refuses_mismatched_reports():
    import pytest

    from execubench import host

    base = {
        "pte": {"sha256": "a"},
        "tokenizer": {"sha256": "t"},
        "max_new_tokens": 48,
        "executorch": "x",
        "results": [{"prompt": "p", "pieces": ["a"]}],
    }
    with pytest.raises(ValueError, match="no results"):
        host.compare(base, {**base, "results": []})
    with pytest.raises(ValueError, match="caps"):
        host.compare(base, {**base, "max_new_tokens": 64})
    assert host.compare(base, dict(base))["identical"]


def test_repeat_check_backs_the_repeatability_sentence():
    import json

    folder = ROOT / "data" / "runtime" / "compat-1.4.0-vs-1.5.1" / "repeat-check"
    reports = sorted(folder.glob("*.json"))
    assert len(reports) == 4
    for path in reports:
        report = json.loads(path.read_text())
        assert all(r["repeats"] == 2 and r["repeats_identical"] for r in report["results"]), path.name
