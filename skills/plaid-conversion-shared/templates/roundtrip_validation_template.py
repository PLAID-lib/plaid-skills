"""Template: read a PLAID dataset from disk and validate roundtrip assumptions."""

from __future__ import annotations

from pathlib import Path

from plaid.storage import init_from_disk


def main() -> None:
    local_dir = Path("TODO_DATASET_DIR")
    split = "train"

    required_scalars = ["TODO_SCALAR"]
    required_feature_paths = ["TODO_BASE/Zone/VertexFields/TODO_FIELD"]

    datasetdict, converterdict = init_from_disk(local_dir)
    dataset = datasetdict[split]
    converter = converterdict[split]

    # Check only a subset first for fast feedback
    n_check = min(5, len(dataset))

    for idx in range(n_check):
        sample = converter.to_plaid(dataset, idx)

        # Scalars check
        scalar_names = set(sample.get_scalar_names())
        for scalar in required_scalars:
            if scalar not in scalar_names:
                raise ValueError(f"Missing scalar '{scalar}' in sample idx={idx}")

        # Feature-path checks (first mesh time)
        times = sample.get_all_time_values()
        if len(times) == 0:
            raise ValueError(f"No mesh times found for sample idx={idx}")

        t0 = times[0]
        for path in required_feature_paths:
            value = sample.get_feature_by_path(path=path, time=t0)
            if value is None:
                raise ValueError(
                    f"Missing feature path '{path}' for sample idx={idx}, time={t0}"
                )

    print(f"Roundtrip validation passed for {n_check} sample(s) in split '{split}'.")


if __name__ == "__main__":
    main()
