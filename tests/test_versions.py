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
    assert (folder / "SUMMARY.md").read_text() == host.summary_markdown(folder)
    for compare in folder.glob("*.compare.json"):
        stem = compare.name.removesuffix(".compare.json")
        a = (folder / f"{stem}.executorch-1.4.0.json").read_text()
        b = (folder / f"{stem}.executorch-1.5.1.json").read_text()
        import json

        assert json.loads(compare.read_text()) == host.compare(json.loads(a), json.loads(b))
