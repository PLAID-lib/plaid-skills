"""Template: save converted samples to PLAID storage backend.

Replace TODO values, then run as a script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plaid import Sample
from plaid.storage import save_to_disk


def sample_constructor(id_: Any) -> Sample:
    """TODO: implement project-specific conversion to Sample."""
    raise NotImplementedError("Implement sample_constructor before running save_to_disk")


def main() -> None:
    output_folder = Path("TODO_OUTPUT_FOLDER")

    # TODO: replace with real ids
    ids = {
        "train": ["TODO_TRAIN_ID_1", "TODO_TRAIN_ID_2"],
        "test": ["TODO_TEST_ID_1"],
    }

    infos = {
        "legal": {
            "owner": "TODO_OWNER",
            "license": "TODO_LICENSE",
        },
        "data_production": {
            "type": "TODO_TYPE",  # simulation / experiment / etc.
            "physics": "TODO_PHYSICS",
        },
    }

    save_to_disk(
        output_folder=output_folder,
        sample_constructor=sample_constructor,
        ids=ids,
        backend="hf_datasets",  # or "zarr" / "cgns"
        infos=infos,
        pb_defs=None,
        num_proc=1,
        verbose=True,
        overwrite=False,
    )


if __name__ == "__main__":
    main()
