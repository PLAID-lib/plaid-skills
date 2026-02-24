---
name: plaid-conversion
description: This skill helps users convert scientific datasets to the PLAID format. It provides practical conversion knowledge through real-world examples, documented patterns, and explicit guidance on handling diverse dataset structures.
---

# PLAID Dataset Conversion Skill

## Overview

This skill helps users convert scientific datasets to the PLAID format. It provides practical conversion knowledge through real-world examples, documented patterns, and explicit guidance on handling diverse dataset structures.

---

## Skill Purpose

**Primary Goal**: Assist users in converting heterogeneous scientific datasets to PLAID format while preserving scientific semantics.

**Target Users**:
- Researchers converting their datasets to PLAID
- Data engineers working with scientific simulations
- ML practitioners preparing physics-based datasets
- Tool developers building PLAID conversion pipelines

---

## Core Capabilities

### What This Skill Provides

1. **Working Conversion Examples** (`examples/conversions/`)
   - Real, production-tested conversion scripts for diverse datasets
   - Coverage of static/temporal, structured/unstructured, nodal/cell-centered data
   - Dataset-specific implementations that prioritize correctness over generality

2. **Pattern Documentation** (`examples/patterns/`)
   - Recurring semantic choices in dataset conversion
   - Guidance on temporal structure, field locations, trajectory grouping
   - Cross-references to example implementations

3. **Conversion Guidance**
   - How to choose the right conversion approach
   - Common pitfalls and design decisions
   - Authority hierarchy for resolving conflicts

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

## Key Principles

### What Makes PLAID Conversions Unique

1. **Dataset-Specific by Design**
   - Conversion scripts encode domain knowledge explicitly
   - Not intended to be generic or fully reusable
   - Preserve scientific semantics over code uniformity

2. **Explicit Over Implicit**
   - Make assumptions visible in code
   - Document non-obvious semantic choices
   - Avoid hiding complexity in abstractions

3. **Correctness Over Convenience**
   - Preserve temporal structure, field locations, trajectory grouping
   - Never silently change scientific meaning
   - Validate on small subsets before full conversion

---

## How to Use This Skill

### For Converting Your Dataset

1. **Find the closest example** in `examples/conversions/`
   - Static or temporal?
   - Structured or unstructured mesh?
   - Nodal or cell-centered fields?

2. **Read relevant patterns** in `examples/patterns/`
   - `static_vs_temporal_samples.md` - Independent states vs time evolution
   - `trajectory_datasets.md` - One sample per physical trajectory
   - `external_time_metadata.md` - Deriving time from sidecar files
   - `nodal_vs_cell_fields.md` - Preserving field location semantics

3. **Adapt to your dataset**
   - Update paths and data sources
   - Modify mesh/tree construction
   - Adjust metadata mapping
   - Keep semantic choices explicit

4. **Validate incrementally**
   - Test on a small subset first
   - Verify scientific correctness
   - Scale up only after validation

### Quick Reference Map

| Dataset Type | Example Script | Key Patterns |
|---|---|---|
| Static CFD data | `drivaerml.py` | Static samples, cell fields |
| Temporal simulations | `force_asr.py`, `thewell_turbulent_layer_2d.py` | Trajectories, external time |
| Parameterized PDEs | `pdebench_2d_darcy_flow.py` | Static samples, structured grids |
| Geometry datasets | `shapenetcar.py` | Static samples, unstructured meshes |

---

## Assistant Behavior Guidelines

### Do:
- **Explain patterns** from existing examples rather than generating full scripts
- **Point to relevant examples** that match user's dataset characteristics
- **State assumptions explicitly** when inferring dataset semantics
- **Highlight common pitfalls** documented in patterns
- **Respect scientific semantics** in all suggestions

### Don't:
- **Auto-generate full conversion scripts** without understanding the dataset
- **Invent PLAID APIs** or abstractions not in the codebase
- **Hide uncertainty** or guess missing semantics
- **Suggest refactors** that alter scientific meaning
- **Treat scripts as generic templates** - they are dataset-specific

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

## Common Conversion Patterns

### Static vs Temporal Samples

**Static**: One `Sample` = one independent configuration
- No time loop
- Single `sample.add_tree()` call
- Examples: `shapenetcar.py`, `pdebench_2d_darcy_flow.py`

**Temporal**: One `Sample` = one physical trajectory
- Loop over time steps
- Multiple `sample.add_tree(time=...)` calls
- Examples: `force_asr.py`, `thewell_turbulent_layer_2d.py`

### Field Location Semantics

**Nodal fields**: Values defined at mesh vertices
- Most common in structural and fluid simulations
- Examples: `shapenetcar.py`, `force_asr.py`

**Cell-centered fields**: Values defined on elements
- Common in finite volume methods
- Example: `pdebench_2d_darcy_flow.py`

### External Metadata

Some datasets require parsing external files for temporal structure:
- PVD/XML files for time step information
- CSV files for parameter values
- Example: `force_asr.py` parses `.pvd` for time values

---

## Dependencies

Conversion scripts may require external libraries not in PLAID core:
- `Muscat` - mesh manipulation
- `h5py`, `plyfile` - file parsing
- `pandas` - metadata handling
- VTK/OpenFOAM tooling - domain-specific parsers

**Important**: These are needed to *convert* datasets, not to *use* already-converted PLAID datasets.

---

## When to Use This Skill

✅ **Use when**:
- Converting a scientific dataset to PLAID
- Understanding PLAID conversion patterns
- Debugging conversion scripts
- Learning from real-world examples

❌ **Don't use for**:
- Generic PLAID tutorials (see main docs)
- Loading/using existing PLAID datasets
- Questions about PLAID core APIs

---

## Related Resources

- **PLAID Documentation**: https://plaid-lib.readthedocs.io/en/stable/
- **PLAID Source Code**: https://github.com/PLAID-lib/plaid
- **Contributing Guide**: See `../CONTRIBUTING_CONVERSIONS.md`
- **Conversion Template**: See `examples/conversions/_template.py`

---

## Notes

This skill is tool- and model-agnostic. It applies to:
- IDE assistants
- Chat-based assistants  
- Automated analysis tools
- Any LLM-based helper working with this repository

The guidance prioritizes scientific correctness and practical applicability over code elegance or generic abstraction.
