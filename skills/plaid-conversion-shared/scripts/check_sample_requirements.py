"""Quick validation helper for one PLAID Sample.

Usage:
    python check_sample_requirements.py

Then adapt REQUIRED_SCALARS and REQUIRED_FEATURE_PATHS for your project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from plaid import Sample

# TODO: import your actual constructor
# from my_conversion_module import sample_constructor


REQUIRED_SCALARS = ["TODO_SCALAR"]
REQUIRED_FEATURE_PATHS = ["TODO_BASE/Zone/VertexFields/TODO_FIELD"]


def _assert_sample_requirements(sample: Sample) -> None:
    scalar_names = set(sample.get_scalar_names())
    for scalar in REQUIRED_SCALARS:
        if scalar not in scalar_names:
            raise ValueError(f"Missing required scalar: {scalar}")

    times = sample.get_all_time_values()
    if len(times) == 0:
        raise ValueError("Sample has no mesh time values")

    t0 = times[0]
    for path in REQUIRED_FEATURE_PATHS:
        value = sample.get_feature_by_path(path=path, time=t0)
        if value is None:
            raise ValueError(f"Missing required feature path: {path} at time={t0}")


def run_check(sample_constructor: Callable[[Any], Sample], sample_id: Any) -> None:
    sample = sample_constructor(sample_id)
    print(sample)
    print(sample.summarize())
    print(sample.check_completeness())
    _assert_sample_requirements(sample)
    print("Sample requirement checks passed.")


if __name__ == "__main__":
    raise SystemExit(
        "Import run_check(...) from your own script, or edit this file to wire your sample_constructor."
    )
