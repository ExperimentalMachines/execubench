"""Validate generated files against the JSON Schemas in schemas/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"


def validator(name: str):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    docs = {p.name: json.loads(p.read_text()) for p in SCHEMAS.glob("*.schema.json")}
    resources = []
    for file_name, doc in docs.items():
        res = Resource.from_contents(doc)
        # Reachable both by $id and by the relative file name used in "$ref".
        resources += [(doc["$id"], res), (file_name, res)]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(docs[name], registry=registry, format_checker=Draft202012Validator.FORMAT_CHECKER)


def errors_for(name: str, instance: dict) -> list[str]:
    v = validator(name)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(instance)]


def validate_repo() -> list[str]:
    """Every generated file that has a schema, checked. Returns readable error lines."""
    out = []
    devices = ROOT / "data" / "devices" / "devices.json"
    if devices.exists():
        for rec in json.loads(devices.read_text())["devices"]:
            out += [f"devices.json {rec.get('id')}: {e}" for e in errors_for("device.schema.json", rec)]
    for example in sorted((SCHEMAS / "examples").glob("*.json")):
        schema = example.name.split(".")[0] + ".schema.json"
        out += [f"{example.name}: {e}" for e in errors_for(schema, json.loads(example.read_text()))]
    return out
