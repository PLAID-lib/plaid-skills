# Example Skill Usage

This document demonstrates how an AI assistant should use the PLAID conversion skill to help a user.

---

## Example Interaction 1: Static Geometry Dataset

### User Request
> I need to convert a dataset of 3D car meshes with aerodynamic pressure fields. Each mesh is independent, stored as STL files with pressure values in CSV format.

### Assistant Response (Following Skill Guidelines)

Based on your description, this is a **static dataset** with unstructured meshes. The closest example in this skill is **`shapenetcar.py`**, which handles:
- Static triangular meshes
- Nodal scalar fields
- Independent samples (no temporal evolution)

**Key patterns to apply:**

1. **Static Sample Semantics** (`patterns/static_vs_temporal_samples.md`)
   - One `Sample` = one car geometry
   - No time loop needed
   - Single `sample.add_tree()` call per mesh

2. **Mesh Construction**
   - Parse STL files to extract vertices and connectivity
   - Map CSV pressure values to mesh nodes
   - Use `Muscat` or similar library for mesh handling

**Conversion outline:**
```python
def generate_samples():
    for car_id, stl_path in enumerate(stl_files):
        # Create new sample
        sample = Sample(identifier=f"car_{car_id}")
        
        # Load mesh from STL
        mesh = load_stl(stl_path)
        
        # Load pressure data from CSV
        pressure = load_pressure_csv(car_id)
        
        # Add mesh tree with pressure field
        sample.add_tree(
            support=mesh,
            fields={"pressure": pressure},
            space="physical"
        )
        
        yield sample
```

**Next steps:**
1. Review `examples/conversions/shapenetcar.py` for detailed implementation
2. Define your `ProblemDefinition` with input/output features
3. Set up metadata (data source, license) in `infos`
4. Test on a small subset before full conversion

---

## Example Interaction 2: Temporal Simulation Dataset

### User Request
> I have CFD simulations of turbulent flow. Each simulation has 100 time steps stored in separate VTK files. The time values are in a `.pvd` metadata file.

### Assistant Response (Following Skill Guidelines)

This is a **temporal trajectory dataset**. The closest examples are:
- **`force_asr.py`** - Shows external time metadata parsing from `.pvd`
- **`thewell_turbulent_layer_2d.py`** - Demonstrates temporal tree construction

**Key patterns to apply:**

1. **Temporal Sample Semantics** (`patterns/static_vs_temporal_samples.md`)
   - One `Sample` = one complete simulation trajectory
   - Loop over time steps
   - Multiple `sample.add_tree(time=...)` calls

2. **External Time Metadata** (`patterns/external_time_metadata.md`)
   - Parse `.pvd` file to extract time values
   - Align time values with VTK file indices
   - Handle potential subsampling

**Critical considerations:**

⚠️ **Common pitfall**: Don't create separate samples for each time step. All time steps of one simulation belong to a single `Sample` object.

**Conversion outline:**
```python
def generate_samples():
    for sim_id in simulation_ids:
        sample = Sample(identifier=f"sim_{sim_id}")
        
        # Parse PVD for time values
        time_values = parse_pvd(f"sim_{sim_id}.pvd")
        
        # Loop over time steps
        for time_idx, time_val in enumerate(time_values):
            # Load VTK file for this time step
            vtk_file = f"sim_{sim_id}/timestep_{time_idx}.vtk"
            mesh, fields = load_vtk(vtk_file)
            
            # Add tree with explicit time
            sample.add_tree(
                time=time_val,
                support=mesh,
                fields=fields,
                space="physical"
            )
        
        yield sample
```

**Reference implementation:**
See `examples/conversions/force_asr.py`, particularly:
- PVD parsing logic (lines ~50-80)
- Time loop structure (lines ~120-150)
- Temporal field attachment

---

## Example Interaction 3: User Asks for Clarification

### User Request
> Should I use nodal or cell-centered fields for my finite volume simulation data?

### Assistant Response (Following Skill Guidelines)

This depends on how your simulation **stores** the data, not on what might be more convenient.

**From `examples/patterns/nodal_vs_cell_fields.md`:**

- **Nodal fields**: Values defined at mesh vertices
  - Common in FEM simulations
  - Use `nodeFields` in PLAID

- **Cell-centered fields**: Values defined on element centers
  - Common in finite volume methods
  - Use `CellData` or `elemFields` in PLAID

**For finite volume methods**, data is typically cell-centered by design. You should preserve this semantic in PLAID rather than interpolating to nodes.

**Example:** `pdebench_2d_darcy_flow.py` uses cell-centered fields:
```python
sample.add_tree(
    support=mesh,
    fields={
        "permeability": permeability,  # Cell-centered
        "pressure": pressure            # Cell-centered
    },
    field_location="element"
)
```

**Important**: Never convert between field locations just for uniformity - this changes the scientific meaning of the data.

---

## Example Interaction 4: Assistant Refuses Overreach

### User Request
> Can you automatically convert my dataset? Here's a zip file with the data.

### Assistant Response (Following Skill Guidelines)

I can **guide** you through the conversion process, but I cannot automatically convert arbitrary datasets. Here's why:

**From SKILL.md principles:**
- Conversion scripts are **intentionally dataset-specific**
- They encode **domain knowledge** that requires human judgment
- Scientific semantics must be preserved (not guessed)

**What I can do:**
1. Help you identify the closest example script
2. Explain relevant patterns for your dataset structure
3. Guide you through key decisions (static vs temporal, field locations, etc.)
4. Review your conversion code for common issues

**What you need to provide:**
- Dataset structure description (mesh type, fields, time semantics)
- Scientific context (what do the fields represent?)
- Desired problem definition (inputs/outputs for ML)

**Next step:** Please fill out `template.md` with your dataset details, and I'll point you to the most relevant examples and patterns.

---

## Key Takeaways

These examples demonstrate the skill's core principle: **Advisory, not automatic**.

The assistant should:
✅ Explain patterns from existing examples
✅ Point to relevant code in the repository
✅ Highlight semantic choices and pitfalls
✅ Guide users through decisions

The assistant should not:
❌ Generate full conversion scripts without context
❌ Make scientific assumptions without stating them
❌ Hide complexity or pretend conversions are trivial
❌ Suggest changes that alter scientific meaning
