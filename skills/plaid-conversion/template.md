# PLAID Conversion Template

Use this template to get guided assistance with converting your dataset to PLAID format.

---

## Dataset Information

**Dataset Name**: [Your dataset name]

**Dataset Type**: 
- [ ] Static (independent samples, no time evolution)
- [ ] Temporal (time-dependent trajectories)

**Data Source**: [URL or reference to raw data]

**License**: [Data license information]

---

## Dataset Structure

### Mesh/Grid Type
- [ ] Structured (regular grid, rectilinear)
- [ ] Unstructured (triangular/tetrahedral mesh)
- [ ] Point cloud
- [ ] Other: _____________

### Field Locations
- [ ] Nodal (values at mesh vertices)
- [ ] Cell-centered (values on elements)
- [ ] Mixed (both nodal and cell fields)

### Temporal Structure (if applicable)
- [ ] Time values embedded in data files
- [ ] Time values in external metadata (e.g., PVD, CSV)
- [ ] Fixed time step size: _____________
- [ ] Variable time steps

---

## Input Data Format

**File Format(s)**: [HDF5, VTK, PLY, OpenFOAM, etc.]

**Required External Libraries**: [e.g., h5py, plyfile, Muscat]

**Data Layout**: 
```
Describe the structure of your input files:
- How are samples organized?
- How are fields named?
- How is time information stored?
```

---

## Scientific Semantics

### Sample Semantics
- [ ] One sample = one geometry/configuration (static)
- [ ] One sample = one complete trajectory (temporal)
- [ ] One sample = one parameter setting
- [ ] Other: _____________

### Features to Include

**Input Features** (what the model receives):
- Field name 1: [description, location, shape]
- Field name 2: [description, location, shape]
- ...

**Output Features** (what the model predicts):
- Field name 1: [description, location, shape]
- Field name 2: [description, location, shape]
- ...

**Constant Features** (metadata/parameters):
- Parameter 1: [description, type]
- Parameter 2: [description, type]
- ...

---

## Conversion Questions

### 1. Dataset Characteristics
Based on your answers above, which example script is closest to your dataset?
- `drivaerml.py` - Static CFD data
- `force_asr.py` - Temporal simulations with external time metadata
- `pdebench_2d_darcy_flow.py` - Parameterized PDEs on structured grids
- `shapenetcar.py` - Static unstructured meshes
- `thewell_turbulent_layer_2d.py` - Temporal trajectories on structured grids

### 2. Specific Challenges
What aspects of your dataset conversion are unclear or challenging?
- [ ] Time structure and trajectory grouping
- [ ] Field location mapping (nodal vs cell)
- [ ] Mesh construction from raw data
- [ ] Parameter/metadata handling
- [ ] Performance/memory for large datasets
- [ ] Other: _____________

### 3. Assistance Needed
What would be most helpful?
- [ ] Explain patterns from similar examples
- [ ] Guide through mesh construction approach
- [ ] Clarify temporal semantics
- [ ] Help with problem definition structure
- [ ] Review existing conversion script
- [ ] Other: _____________

---

## Next Steps

Once you've filled this out, the assistant can:
1. Point you to the most relevant conversion example
2. Identify applicable patterns from `patterns/`
3. Explain key decisions for your specific dataset
4. Guide you through the conversion process step by step

Remember: The goal is to create a **dataset-specific** conversion script that preserves scientific correctness, not a generic converter.
