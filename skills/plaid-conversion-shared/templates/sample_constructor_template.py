"""Template: build one PLAID Sample from a user sample identifier.

Replace TODO sections with project-specific logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plaid import Sample


def sample_constructor(id_: Any) -> Sample:
    """Return one PLAID Sample for a single user-provided identifier.

    Args:
        id_: Sample identifier. Can be int/path/tuple/string depending on user data model.
    """
    # TODO: resolve input paths from id_
    sample_dir = Path(str(id_))

    sample = Sample()

    # --- Scalars ------------------------------------------------------------
    # TODO: replace with real scalar loading logic
    # Example:
    # sample.add_scalar("rpm", float(...))

    # --- Mesh / Tree --------------------------------------------------------
    # TODO: if user has CGNS tree ready, add directly:
    # sample.add_tree(cgns_tree)
    # Otherwise, load/build nodes/elements then set in SampleFeatures pipeline.

    # --- Fields -------------------------------------------------------------
    # TODO: replace with real field loading logic
    # Example:
    # pressure = np.load(sample_dir / "pressure.npy")
    # sample.add_field(
    #     name="pressure",
    #     field=pressure,
    #     location="Vertex",
    #     base_name="Base",
    #     zone_name="Zone",
    # )

    return sample


if __name__ == "__main__":
    # quick smoke-test on one sample id
    test_id = "TODO_SAMPLE_ID"
    s = sample_constructor(test_id)
    print(s)
    print(s.summarize())
    print(s.check_completeness())
