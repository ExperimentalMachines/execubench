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
