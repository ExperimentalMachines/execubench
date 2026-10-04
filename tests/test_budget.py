"""docs/PLAN.md section 7.2 quotes execubench/budget.py; the table rows must match exactly."""

import re
from pathlib import Path

from execubench import budget

PLAN_ALL = (Path(__file__).resolve().parent.parent / "docs" / "PLAN.md").read_text()
# Only section 7.2: the grid table in 7.1 reuses some of the same row labels.
PLAN = PLAN_ALL[PLAN_ALL.index("### 7.2 Cost model") : PLAN_ALL.index("## 8.")]
ROWS = {
    "quality reference": "Quality reference",
    "agreement": "Agreement",
    "speed": "Speed",
    "window sweep": "Window sweep",
    "long context": "Long context",
}


def _row(label: str) -> list[str]:
    m = re.search(rf"^\| {re.escape(label)} \|([^\n]*)$", PLAN, flags=re.M)
    assert m, label
    return [c.strip() for c in m.group(1).split("|") if c.strip()]


def test_plan_table_rows_match_the_calculator():
    b = budget.budget()
    for key, label in ROWS.items():
        assert _row(label) == [f"{b['minutes'][key]:,}", f"{b['jobs'][key]:,}"], label
    assert _row("Job overhead") == [f"{b['minutes']['job overhead']:,}"]
    assert _row("**Total before contingency**") == [
        f"**{b['minutes_before_contingency']:,}**",
        f"**{sum(b['jobs'].values()):,}**",
    ]
    assert _row("**With 20 percent contingency**")[0].startswith(f"**{b['minutes_with_contingency']:,}**")
    assert f"**${round(b['metered_usd'], -2):,}**" in PLAN
    assert f"**${b['unmetered_usd_at_70pct_use']:,}**" in PLAN


def test_jobs_never_mix_models_or_devices():
    a = budget.Assumptions()
    # Each model on each Tier A flagship needs ceil(700 / 120) = 6 jobs; on the A56, ceil(2100 / 120) = 18.
    assert budget.budget(a)["jobs"]["quality reference"] == a.models * (3 * 6 + 1 * 18)
