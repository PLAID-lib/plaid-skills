# plaid-skills

Practical conversion knowledge for the [PLAID](https://github.com/PLAID-lib) ecosystem.

This repository is a curated, code-first reference for converting scientific datasets into the PLAID format. It contains:

- **real conversion scripts** (dataset-specific and production-oriented),
- **pattern notes** extracted from those scripts,
- and **agent guidance** describing how assistants should reason about these materials.

The goal is to help users and tools understand *how PLAID is used in practice* for heterogeneous scientific datasets.

---

## 10-minute quick start

If you are new to this repo, use this minimal path:

1. Open [`skills/examples/conversions/README.md`](./skills/examples/conversions/README.md).
2. Pick the script closest to your dataset (static/temporal, structured/unstructured, nodal/cell-centered).
3. Check required external dependencies and raw-data source for that script.
4. Set all placeholder paths/repo IDs (look for assertions on `/path/to/...` and `channel/repo`).
5. Run the script on a small subset first, verify semantic correctness, then scale up.

For contribution rules and a starter scaffold, see:

- [`CONTRIBUTING_CONVERSIONS.md`](./CONTRIBUTING_CONVERSIONS.md)
- [`skills/examples/conversions/_template.py`](./skills/examples/conversions/_template.py)

---

## Repository structure

```text
.
├── AGENT.md
├── LICENSE
└── skills
    ├── examples
    │   └── conversions
    │       ├── README.md
    │       ├── drivaerml.py
    │       ├── force_asr.py
    │       ├── pdebench_2d_darcy_flow.py
    │       ├── shapenetcar.py
    │       └── thewell_turbulent_layer_2d.py
    └── patterns
        ├── external_time_metadata.md
        ├── nodal_vs_cell_fields.md
        ├── static_vs_temporal_samples.md
        └── trajectory_datasets.md
```

---

## What this repo is (and is not)

### ✅ What it is

- A **skills library** for dataset conversion to PLAID.
- A **reference set of working examples** used on real datasets.
- A **semantic guide** to recurring choices (time, trajectories, field locations, etc.).

### ❌ What it is not

- Not a generic one-click converter for arbitrary datasets.
- Not a stable Python package with reusable APIs.
- Not a minimal PLAID tutorial.

Most scripts are intentionally explicit and dataset-specific.

---

## Authority and reading order

As defined in `AGENT.md`, interpretation priority is:

1. **Conversion examples** in `skills/examples/conversions/` (authoritative in practice)
2. **Pattern documents** in `skills/patterns/`
3. Conceptual PLAID docs
4. Source-level API details

If conceptual docs and examples differ, examples win for practical conversion behavior.

---

## Conversion examples included

The scripts in `skills/examples/conversions/` cover multiple dataset families and semantics:

- **DrivAerML** (`drivaerml.py`)
  - Steady-state automotive CFD
  - Static samples (no time axis)
  - OpenFOAM meshes/fields + CSV metadata
  - Demonstrates placeholders for dataset-specific parser integration

- **ForceASR** (`force_asr.py`)
  - Time-dependent phase-field fracture simulations
  - One sample = one trajectory
  - Time values read from external metadata (`.pvd`)
  - Mixed nodal fields + time-varying global quantities

- **PDEBench 2D Darcy Flow** (`pdebench_2d_darcy_flow.py`)
  - Static, parameterized PDE dataset
  - Structured rectilinear mesh reused across samples
  - Cell-centered fields (`CellData`)
  - Script-level parameter sweep (`beta` values)

- **ShapeNet-Car** (`shapenetcar.py`)
  - Static triangular meshes with nodal scalar fields
  - Unstructured geometry conversion
  - Sequential and parallel generation variants
  - Multi-backend export (`hf_datasets`, `cgns`, `zarr`)

- **The Well: Turbulent Radiative Layer 2D** (`thewell_turbulent_layer_2d.py`)
  - Temporal trajectories on structured grids
  - Per-time tree/field assembly
  - Boundary tags and trajectory-aware sample construction

---

## Patterns documented

Pattern notes in `skills/patterns/` summarize recurring semantics and pitfalls:

- `static_vs_temporal_samples.md` — when to model independent states vs time evolution
- `trajectory_datasets.md` — one sample per physical trajectory
- `external_time_metadata.md` — deriving time from sidecar metadata (PVD/XML/etc.)
- `nodal_vs_cell_fields.md` — preserving field location semantics (node vs element)

These documents are especially useful when adapting an existing script to a new dataset.

### Pattern-to-example map

- Static sample semantics: `drivaerml.py`, `shapenetcar.py`, `pdebench_2d_darcy_flow.py`
- Temporal trajectory semantics: `force_asr.py`, `thewell_turbulent_layer_2d.py`
- External time metadata: `force_asr.py` (`.pvd` parsing)
- Nodal vs cell-centered fields:
  - Nodal: `shapenetcar.py`, `force_asr.py`, `thewell_turbulent_layer_2d.py`
  - Cell-centered: `pdebench_2d_darcy_flow.py` (and likely `drivaerml.py` depending on parser mapping)

---

## Typical usage workflow

1. **Find the closest conversion script** to your dataset characteristics.
2. **Read matching pattern docs** for semantic choices you must preserve.
3. **Adapt paths, parsers, and metadata mapping** (never assume generic portability).
4. **Keep semantics explicit** in `Sample`, tree, field, and time construction.
5. **Export with desired backend(s)** and optionally publish to the Hub.

> Most scripts include placeholder path assertions (`/path/to/...`) to force explicit user configuration.

---

## Dependencies

This repo intentionally includes scripts that may require **external, dataset-specific dependencies** not part of PLAID’s core dependency set (for example: `Muscat`, `plyfile`, `h5py`, `pandas`, VTK/OpenFOAM tooling).

Important distinctions:

- Extra dependencies are often needed to **convert** raw datasets.
- They are not necessarily needed to **consume** already-converted PLAID datasets.

Check imports at the top of each conversion script before running it.

---

## Guidance for assistant/tool authors

If you build an assistant around this repo, align behavior with `AGENT.md`:

- Prefer explaining existing patterns over auto-generating full conversions.
- Do not invent PLAID APIs or hide uncertain assumptions.
- Keep scientific semantics intact; avoid style-driven refactors.
- Treat conversion scripts as dataset-specific artifacts, not generalized templates.

---

## Contributing

When adding a new conversion skill:

- Add one script per dataset in `skills/examples/conversions/`.
- Keep dataset assumptions explicit in code.
- Avoid introducing PLAID APIs for one-off needs.
- Preserve scientific meaning over normalization of code style.

Optional but encouraged:

- Add a short module docstring describing dataset structure.
- Add or update pattern notes when a new recurring semantic appears.

Before opening a PR, review:

- [`CONTRIBUTING_CONVERSIONS.md`](./CONTRIBUTING_CONVERSIONS.md)
- [`skills/examples/conversions/_template.py`](./skills/examples/conversions/_template.py)

---

## License

This repository is distributed under the **BSD 3-Clause License**.
See [`LICENSE`](./LICENSE) for full terms.
