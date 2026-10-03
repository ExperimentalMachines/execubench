"""The v1 device-minute budget, from the grid and stated assumptions.

Every input below is an assumption until P1 measures it; docs/PLAN.md section 7 quotes the
output of `python -m execubench budget` and must be regenerated when an input changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

PRICE_PER_MINUTE = 0.17  # USD, metered, https://aws.amazon.com/device-farm/pricing/
SLOT_PER_MONTH = 250.0  # USD, one unmetered Android slot
SLOT_HOURS_PER_MONTH = 720  # a slot is a concurrency of one, around the clock at best


@dataclass
class Assumptions:
    models: int = 16
    quality_generations: int = 2800  # full v1 selection per model, provisional until P2 freezes it
    agreement_generations: int = 300  # first 40 IDs per dataset and mode
    seconds_per_generation_flagship: float = 15.0
    midrange_slowdown: float = 3.0
    speed_minutes_flagship: float = 15.0  # one speed track per model and device
    long_context_minutes: float = 20.0  # one long-context file on one device
    job_overhead_minutes: float = 8.0  # identify, stage, start, cool down, warm up
    usable_minutes_per_job: float = 120.0  # 150-minute limit less a 20 percent margin
    retry_contingency: float = 0.20
    # Tiers, as (flagship-class count, mid-range count), from data/devices/standard.yaml.
    quality_devices: tuple[int, int] = (3, 1)  # Tier A: S25 Ultra, Pixel 10, Tab S11; A56
    agreement_devices: tuple[int, int] = (4, 3)  # A+ and C flagships; Tier B
    speed_devices: tuple[int, int] = (7, 4)
    window_sweep_files: int = 13  # SmolLM2-360M 4 windows, Qwen3-1.7B 4, LFM2.5-1.2B 5
    window_sweep_devices: int = 2  # S25 Ultra, Pixel 10
    long_context_files_per_model: int = 2  # the 8k file and the largest published window
    long_context_devices: int = 2  # S25 Ultra, Pixel 10
    rows: list = field(default_factory=list)


def _track(minutes_flagship: float, devices: tuple[int, int], a: Assumptions) -> float:
    flagship, mid = devices
    return minutes_flagship * (flagship + mid * a.midrange_slowdown)


def budget(a: Assumptions | None = None) -> dict:
    a = a or Assumptions()
    gen_min = a.seconds_per_generation_flagship / 60
    tracks = {
        "quality reference": a.models * _track(a.quality_generations * gen_min, a.quality_devices, a),
        "agreement": a.models * _track(a.agreement_generations * gen_min, a.agreement_devices, a),
        "speed": a.models * _track(a.speed_minutes_flagship, a.speed_devices, a),
        "window sweep": a.window_sweep_files * a.window_sweep_devices * a.speed_minutes_flagship,
        "long context": a.models * a.long_context_files_per_model * a.long_context_devices * a.long_context_minutes,
    }
    jobs = {
        "quality reference": math.ceil(tracks["quality reference"] / a.usable_minutes_per_job),
        "agreement": max(
            a.models * sum(a.agreement_devices), math.ceil(tracks["agreement"] / a.usable_minutes_per_job)
        ),
        "speed": a.models * sum(a.speed_devices),
        "window sweep": a.window_sweep_files * a.window_sweep_devices,
        "long context": a.models * a.long_context_files_per_model * a.long_context_devices,
    }
    overhead = sum(jobs.values()) * a.job_overhead_minutes
    work = sum(tracks.values()) + overhead
    total = work * (1 + a.retry_contingency)
    hours = total / 60
    return {
        "assumptions": {k: v for k, v in vars(a).items() if k != "rows"},
        "minutes": {**{k: round(v) for k, v in tracks.items()}, "job overhead": round(overhead)},
        "jobs": jobs,
        "minutes_before_contingency": round(work),
        "minutes_with_contingency": round(total),
        "device_hours": round(hours),
        "metered_usd": round(total * PRICE_PER_MINUTE),
        "slot_months_at_full_use": round(hours / SLOT_HOURS_PER_MONTH, 1),
        "slot_months_at_70pct_use": round(hours / (SLOT_HOURS_PER_MONTH * 0.7), 1),
        "unmetered_usd_at_70pct_use": round(math.ceil(hours / (SLOT_HOURS_PER_MONTH * 0.7)) * SLOT_PER_MONTH),
    }
