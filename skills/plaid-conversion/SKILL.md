---
name: plaid-conversion
description: Practical guidance for converting scientific datasets to PLAID using real conversion scripts and pattern docs. Use this skill whenever users mention adapting a dataset to PLAID, choosing static vs temporal sample semantics, handling trajectories or external time metadata, mapping nodal vs cell-centered fields, or debugging PLAID conversion logic—even if they do not explicitly ask for this skill by name.
---

# PLAID Dataset Conversion Skill

## Purpose

Help users convert heterogeneous scientific datasets to PLAID **without losing scientific semantics**.

This skill is primarily a **guided reference**: it points to proven conversion scripts and recurring patterns, then helps users adapt those patterns to their own datasets.

## Use This Skill When

Prioritize this skill whenever a user asks about:

- converting a dataset to PLAID,
- adapting or debugging a PLAID conversion script,
- static vs temporal sample structure,
- one-sample-per-trajectory semantics,
- deriving time values from external metadata (`.pvd`, XML, CSV),
- nodal vs cell-centered field mapping,
- preserving scientific meaning while restructuring data.

Do **not** prioritize this skill for generic PLAID usage questions that are unrelated to conversion (e.g., only loading an already-converted PLAID dataset).

## Skill selection and handoff

Use this skill as the **semantic strategy layer** for dataset conversion.

When a user request becomes implementation-specific, route to the specialized skills:

| User intent | Primary skill |
|---|---|
| Decide semantic mapping strategy (static vs temporal, trajectory unit, time metadata alignment, nodal vs cell semantics) | `plaid-conversion` |
| Implement `sample_constructor(id_) -> Sample` from raw files | `plaid-sample-converter` |
| Build `save_to_disk(...)` export pipeline (ids/backends/metadata) | `plaid-storage-writer-pipeline` |
| Load saved dataset and run roundtrip diagnostics with `Converter` | `plaid-storage-reader-roundtrip` |

If a request spans multiple stages, handle the current stage explicitly and propose the next skill handoff.

---

## Authority Hierarchy

When helping with conversions, follow this priority order:

1. **Conversion examples in `examples/conversions/`**
   - These are authoritative for how PLAID is used in practice
   - Real scripts used to generate published datasets

2. **Pattern descriptions in `examples/patterns/`**
   - Document recurring structures and semantics
   - Extracted from actual conversion experience

3. **PLAID documentation** at https://plaid-lib.readthedocs.io/en/stable/
   - Conceptual definitions and API reference
   - May not reflect all practical patterns

4. **Source code** at https://github.com/PLAID-lib/plaid
   - For confirming implementation details

**Resolution Rule**: If conflicts arise, practical examples take precedence over documentation.

---

## Operating Principles

1. **Dataset-specific by design**
   - Conversion scripts should encode domain assumptions explicitly.
   - Do not force generic abstractions when they hide scientific meaning.

2. **Explicit assumptions**
   - State uncertain assumptions directly.
   - Keep semantics visible in code and explanations (time, field location, sample unit).

3. **Correctness over convenience**
   - Never silently alter scientific meaning.
   - Preserve temporal structure, trajectory grouping, and nodal/element semantics.

4. **Incremental validation**
   - Validate on a small subset first.
   - Scale only after semantic correctness is confirmed.

---

## Response Workflow

When this skill triggers, follow this sequence:

1. **Classify dataset semantics**
   - Static vs temporal
   - Structured vs unstructured support
   - Nodal vs cell-centered fields
   - Per-sample unit (configuration, trajectory, parameter setting)

2. **Route to closest conversion example** in `examples/conversions/`

3. **Pull matching pattern docs** in `examples/patterns/`
   - `static_vs_temporal_samples.md`
   - `trajectory_datasets.md`
   - `external_time_metadata.md`
   - `nodal_vs_cell_fields.md`

4. **Provide adaptation guidance**
   - Explicit assumptions
   - Likely pitfalls
   - Suggested structure for mesh/tree/feature mapping

5. **Recommend validation steps**
   - Small subset run
   - Semantic checks before full conversion

### Quick Reference Map

| Dataset Type | Example Script | Key Patterns |
|---|---|---|
| Static CFD data | `drivaerml.py` | Static samples, cell fields |
| Temporal simulations | `force_asr.py`, `thewell_turbulent_layer_2d.py` | Trajectories, external time |
| Parameterized PDEs | `pdebench_2d_darcy_flow.py` | Static samples, structured grids |
| Geometry datasets | `shapenetcar.py` | Static samples, unstructured meshes |

---

## Assistant Behavior Guidelines

### Preferred behaviors

- Explain and reuse proven patterns from existing examples.
- Point to specific files/sections that match the user's dataset shape.
- Keep assumptions explicit, especially where metadata is missing.
- Surface pitfalls early (sample unit mismatch, time alignment, field location drift).
- Keep recommendations faithful to scientific semantics.

### Avoid

- Auto-generating full conversion pipelines from vague prompts.
- Inventing PLAID APIs, wrappers, or abstractions not grounded in repo examples.
- Guessing unstated scientific semantics.
- Refactors that optimize style while changing meaning.
- Presenting scripts as plug-and-play templates.

### Advisory, Not Automatic

This skill positions the assistant as a **knowledgeable guide**, not an automated converter. Users maintain control over scientific decisions.

---

## Dataset Coverage

Current examples cover:

- **DrivAerML** - Steady-state automotive CFD (OpenFOAM)
- **ForceASR** - Time-dependent fracture mechanics
- **PDEBench 2D Darcy Flow** - Parameterized elliptic PDEs
- **ShapeNet-Car** - Static triangular meshes with scalars
- **The Well: Turbulent Radiative Layer 2D** - Temporal fluid dynamics

Each example includes:
- Dataset source and license information
- External dependencies required
- Full working conversion code
- Problem definition and feature mappings

---

## High-Value Pattern Checks

Use these checks proactively when reviewing a user request or script:

### 1) Static vs Temporal sample semantics

- **Static**: one `Sample` per independent state; usually one `add_tree` call.
- **Temporal**: one `Sample` per trajectory; multiple `add_tree(time=...)` calls.

### 2) Field location semantics

- **Nodal** fields live on vertices.
- **Cell-centered** fields live on elements.
- Do not remap location just for convenience.

### 3) External temporal metadata alignment

- Confirm whether file index equals physical time (often false).
- If sidecar metadata exists (`.pvd`, XML, CSV), parse and align explicitly.

---

## Dependencies

Conversion scripts may require external libraries not in PLAID core:
- `Muscat` - mesh manipulation
- `h5py`, `plyfile` - file parsing
- `pandas` - metadata handling
- VTK/OpenFOAM tooling - domain-specific parsers

**Important**: These are needed to *convert* datasets, not to *use* already-converted PLAID datasets.

---

## Related Resources

- **PLAID Documentation**: https://plaid-lib.readthedocs.io/en/stable/
- **PLAID Source Code**: https://github.com/PLAID-lib/plaid
- **Contributing Guide**: See `../../CONTRIBUTING_CONVERSIONS.md`
- **Conversion Template**: See `examples/conversions/_template.py`

---

## Notes

This skill is tool- and model-agnostic. It applies to:
- IDE assistants
- Chat-based assistants  
- Automated analysis tools
- Any LLM-based helper working with this repository

The guidance prioritizes scientific correctness and practical applicability over code elegance or generic abstraction.
