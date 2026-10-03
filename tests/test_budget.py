"""docs/PLAN.md section 7.2 quotes execubench/budget.py; the two must agree."""

from pathlib import Path

from execubench import budget

PLAN = (Path(__file__).resolve().parent.parent / "docs" / "PLAN.md").read_text()


def test_plan_table_matches_the_calculator():
    b = budget.budget()
    for track, minutes in b["minutes"].items():
        assert f"{minutes:,}" in PLAN, track
    for track, jobs in b["jobs"].items():
        assert f"| {jobs:,} |" in PLAN, track
    assert f"**{b['minutes_before_contingency']:,}**" in PLAN
    assert f"**{b['minutes_with_contingency']:,}**" in PLAN
    assert f"about {b['device_hours']:,} device hours" in PLAN
    assert f"**${round(b['metered_usd'], -2):,}**" in PLAN
    assert f"**${b['unmetered_usd_at_70pct_use']:,}**" in PLAN


def test_jobs_never_mix_models_or_devices():
    a = budget.Assumptions()
    b = budget.budget(a)
    # Each model on each Tier A flagship needs ceil(700 / 120) = 6 jobs; on the A56, ceil(2100 / 120) = 18.
    assert b["jobs"]["quality reference"] == a.models * (3 * 6 + 1 * 18)
