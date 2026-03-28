# Raw-to-PLAID mapping table

Fill this table before writing conversion code.

| Raw source path/key | Raw shape/type | PLAID target type | PLAID target name/path | Transform needed | Notes |
|---|---|---|---|---|---|
| `<example: case.csv:rpm>` | `float` | `scalar` | `rpm` | none | one value per sample |
| `<example: pressure.npy>` | `(n_nodes,)` | `field` | `Base/Zone/VertexFields/pressure` | optional reshape | verify node alignment |
| `<example: mesh file>` | mesh object | `tree/nodes/elements` | `sample.add_tree(...)` or `set_nodes(...)` | conversion to CGNS | define location conventions |

## Required checks

- [ ] all required user outputs are mapped
- [ ] each mapped feature has a deterministic loading path
- [ ] field locations and names are explicit
- [ ] unknown mappings are documented as TODO
