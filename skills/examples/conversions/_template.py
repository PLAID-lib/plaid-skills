"""
PLAID Dataset Conversion Example (Template)

Dataset:
- <dataset name>

Purpose:
- Convert <source format / dataset type> to PLAID format.

Key semantics (fill explicitly):
- Sample semantics: <static sample | one trajectory per Sample>
- Time semantics: <none | explicit time axis | external metadata>
- Field location: <nodal | cell-centered | mixed>
- Mesh type: <structured | unstructured>

Notes:
- Keep dataset-specific assumptions explicit.
- Do not turn this into a generic converter framework.
"""

from pathlib import Path

import numpy as np
from plaid import ProblemDefinition, Sample
from plaid.storage import save_to_disk, push_to_hub

# Optional dataset-specific imports (examples):
# import h5py
# import pandas as pd
# from Muscat.Bridges.CGNSBridge import MeshToCGNS


# ------------------------------------------------------------------
# User configuration (required)
# ------------------------------------------------------------------

RAW_DATA_DIR = "/path/to/raw/data"
OUTPUT_DIR = "/path/to/local/output"
REPO_ID = "channel/repo"

assert RAW_DATA_DIR != "/path/to/raw/data", "Please set RAW_DATA_DIR"
assert OUTPUT_DIR != "/path/to/local/output", "Please set OUTPUT_DIR"
assert REPO_ID != "channel/repo", "Please set REPO_ID"

BACKENDS = ["hf_datasets"]  # add "cgns", "zarr" if needed


# ------------------------------------------------------------------
# Dataset metadata
# ------------------------------------------------------------------

infos = {
    "legal": {
        "owner": "<data owner + source URL>",
        "license": "<license>",
    },
    "data_production": {
        "physics": "<physics/domain>",
        "type": "simulation",  # or experiment
        "script": "Converted to PLAID format for standardized access; no changes to data content.",
    },
}


# ------------------------------------------------------------------
# Problem definition
# ------------------------------------------------------------------

constant_features = [
    # e.g. mesh coordinates, connectivity, static globals
]

input_features = [
    # e.g. model inputs / conditioning fields
]

output_features = [
    # e.g. targets
]

pb_def = ProblemDefinition()
pb_def.add_in_features_identifiers(input_features)
pb_def.add_out_features_identifiers(output_features)
pb_def.add_constant_features_identifiers(constant_features)
pb_def.set_task("regression")
pb_def.set_name("regression")
# Optional split examples:
# pb_def.set_train_split({"train": "all"})
# pb_def.set_test_split({"test": "all"})
# pb_def.set_split({"all_samples": list(range(N))})


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------

def generator_fn():
    """
    Yield one PLAID Sample at a time.

    Keep semantics explicit:
    - static: one tree/field set once
    - temporal: loop over time and call add_tree(..., time=t)
    - trajectory: one Sample per trajectory
    """
    data_root = Path(RAW_DATA_DIR)

    # Replace this loop with dataset-specific iteration logic
    for _ in np.arange(0):
        sample = Sample()

        # Example placeholders:
        # tree = MeshToCGNS(mesh, exportOriginalIDs=False)
        # sample.add_tree(tree)
        # sample.add_field("field_name", field_values)
        # sample.add_scalar("global_name", value)

        yield sample


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------

def main():
    generators = {"train": generator_fn}

    for backend in BACKENDS:
        local_folder = f"{OUTPUT_DIR}/{backend}_dataset"

        save_to_disk(
            output_folder=local_folder,
            generators=generators,
            backend=backend,
            infos=infos,
            pb_defs=pb_def,
            overwrite=True,
            verbose=True,
        )

        if backend == "hf_datasets":
            push_to_hub(
                repo_id=REPO_ID,
                local_dir=local_folder,
                viewer=True,
                pretty_name="<dataset pretty name>",
            )


if __name__ == "__main__":
    main()
