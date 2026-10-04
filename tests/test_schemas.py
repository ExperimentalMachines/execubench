"""Schemas compile, examples validate, and the provenance rules bite."""

from execubench import schemas


def test_repo_validates():
    assert schemas.validate_repo() == []


def test_published_value_needs_source_and_date():
    sourced = {"$ref": "common.schema.json#/$defs/sourced"}
    v = schemas.validator("common.schema.json")
    from jsonschema import Draft202012Validator

    check = Draft202012Validator(sourced, registry=v._registry)
    assert list(check.iter_errors({"value": 1, "provenance": "published"}))
    assert not list(
        check.iter_errors({"value": 1, "provenance": "published", "source": "https://x", "retrieved": "2026-10-04"})
    )


def test_unknown_value_must_be_null():
    from jsonschema import Draft202012Validator

    v = schemas.validator("common.schema.json")
    check = Draft202012Validator({"$ref": "common.schema.json#/$defs/sourced"}, registry=v._registry)
    assert list(check.iter_errors({"value": 3.2, "provenance": "unknown"}))
    assert not list(check.iter_errors({"value": None, "provenance": "unknown"}))


import copy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "examples"


def _example(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text())


def _drop(d: dict, path: str) -> None:
    *parents, last = path.split(".")
    for p in parents:
        d = d[p]
    d.pop(last)


def _set(d: dict, path: str, value) -> None:
    *parents, last = path.split(".")
    for p in parents:
        d = d[p]
    d[last] = value


@pytest.mark.parametrize(
    "change",
    [
        ("set", "timings.runner", {}),
        ("drop", "timings.monotonic", None),
        ("drop", "item.prompt_text", None),
        ("set", "memory", {"rss_sampled_peak_mib": None, "rss_mean_mib": None, "samples": 0}),
        (
            "set",
            "state",
            {
                "battery_temp_c_start": None,
                "battery_temp_c_end": None,
                "thermal_status_max": None,
                "thermal_event": False,
                "clock_drop": None,
            },
        ),
    ],
)
def test_a_successful_request_without_evidence_is_rejected(change):
    """Review round 2 found these exact records validating; each must now fail."""
    rec = copy.deepcopy(_example("request.example.json"))
    kind, path, value = change
    _drop(rec, path) if kind == "drop" else _set(rec, path, value)
    assert schemas.errors_for("request.schema.json", rec)


def test_thermal_status_needs_a_value_or_a_reason():
    rec = copy.deepcopy(_example("request.example.json"))
    rec["state"]["thermal_status_max"] = None
    assert schemas.errors_for("request.schema.json", rec)
    rec["state"]["null_reason"] = "thermalservice_absent_before_android_10"
    assert schemas.errors_for("request.schema.json", rec) == []


def test_missing_measurements_are_allowed_with_a_reason():
    rec = copy.deepcopy(_example("request.example.json"))
    rec["memory"] = {
        "rss_sampled_peak_mib": None,
        "rss_mean_mib": None,
        "samples": 0,
        "null_reason": "proc_status_unreadable",
    }
    assert schemas.errors_for("request.schema.json", rec) == []


def test_speed_records_need_their_workload():
    summary = copy.deepcopy(_example("summary.example.json"))
    summary.pop("decode_target")
    assert schemas.errors_for("summary.schema.json", summary)
    req = copy.deepcopy(_example("request.example.json"))
    req["track"] = "speed"
    assert schemas.errors_for("request.schema.json", req)
    req["item"].update({"prompt_bucket": 512, "decode_target": 64})
    assert schemas.errors_for("request.schema.json", req) == []


def test_search_tool_mode_ties_retention_to_the_call():
    req = copy.deepcopy(_example("request.example.json"))
    req["item"]["dataset"] = "retrievalqa/search-tool"
    req["item"].update({"tool_called": True, "answer_in_context": None, "retained_passages": None})
    assert schemas.errors_for("request.schema.json", req)
    req["item"].update({"answer_in_context": False, "retained_passages": ["0" * 64]})
    assert schemas.errors_for("request.schema.json", req) == []
    req["item"].update({"tool_called": False})
    assert schemas.errors_for("request.schema.json", req)
    req["item"].update({"answer_in_context": None, "retained_passages": []})
    assert schemas.errors_for("request.schema.json", req) == []


def test_retrievalqa_requests_record_what_context_was_given():
    req = copy.deepcopy(_example("request.example.json"))
    req["item"]["dataset"] = "retrievalqa/with-context"
    assert schemas.errors_for("request.schema.json", req)
    req["item"].update({"answer_in_context": None, "retained_passages": None})
    assert schemas.errors_for("request.schema.json", req)
    req["item"].update({"answer_in_context": True, "retained_passages": ["0" * 64]})
    assert schemas.errors_for("request.schema.json", req) == []
    summary = copy.deepcopy(_example("summary.example.json"))
    summary["track"], summary["dataset"] = "quality", "retrievalqa/with-context"
    summary.pop("prompt_bucket"), summary.pop("decode_target")
    summary["quality"] = {"answer_in_context": {}}
    assert schemas.errors_for("summary.schema.json", summary)
