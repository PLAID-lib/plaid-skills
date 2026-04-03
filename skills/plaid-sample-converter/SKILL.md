---
name: plaid-sample-converter
description: Convert custom user datasets into PLAID `Sample` objects before storage export. Use this skill whenever a user asks to map raw data (CSV/NumPy/HDF5/mesh outputs/solver exports) into `Sample` scalars, fields, nodes, or trees, even if they do not explicitly say “PLAID Sample”. Also use when they ask for a `sample_constructor(id_) -> Sample`, feature mapping, or pre-export completeness checks. Do not use this skill for `save_to_disk` pipeline authoring or post-export readback debugging.
---

# plaid-sample-converter

Build robust conversion logic from user-owned data into PLAID `Sample` objects.

## Scope

Focus on:

- `plaid.containers.sample.Sample`
- feature population through `add_scalar`, `add_field`, `set_nodes`, `add_tree`
- pre-export completeness checks via `sample.summarize()` and `sample.check_completeness()`

Do **not** drift into unrelated PLAID modules unless needed for minimal context.

## Skill boundaries and handoffs

Use this skill for **in-memory sample construction** from raw data.

When the user moves to storage/export or readback tasks, hand off explicitly:

- For `save_to_disk(...)`, split IDs, backend choice, and write scripts → use `plaid-storage-writer-pipeline`.
- For loading from disk/hub and roundtrip checks with `Converter` → use `plaid-storage-reader-roundtrip`.
- For high-level semantic decisions (trajectory semantics, external time metadata, nodal vs cell field policy) before coding details → use `plaid-conversion`.

## Inputs to collect from user

Collect enough information before coding:

1. raw data format(s) and paths
2. sample identity key (`id_` = index, filename, tuple, etc.)
3. mapping between raw variables and PLAID targets
4. expected mesh representation (existing CGNS tree vs nodes/elements/fields assembly)
5. expected scalar/field names and units

If info is incomplete, produce a mapping proposal and mark unknown fields with TODO placeholders.

## Conversion workflow

1. **Profile the source structure**
   - list candidate variables per sample
   - identify constants vs per-sample features
2. **Produce an explicit mapping table**
   - use `../plaid-conversion-shared/templates/mapping_table_template.md`
3. **Generate `sample_constructor(id_) -> Sample`**
   - use `../plaid-conversion-shared/templates/sample_constructor_template.py`
   - keep loading/parsing deterministic
4. **Add quality checks**
   - print `sample.summarize()`
   - run `sample.check_completeness()`
   - verify critical fields are present and shape-consistent
   - optionally run `../plaid-conversion-shared/scripts/check_sample_requirements.py`
5. **Return a runnable snippet + usage notes**

## Response protocol

1. Ask targeted clarifying questions when raw schema, sample identity, or feature locations are ambiguous.
2. If key inputs are missing, provide a concrete mapping proposal with explicit TODO placeholders rather than guessing.
3. Fail fast on scientifically risky assumptions (e.g., silently converting cell-centered values to nodal values).
4. Keep constructor logic deterministic and easy to debug.

## Output contract

Always return:

1. mapping table (raw source → PLAID feature)
2. final `sample_constructor(id_) -> Sample`
3. short validation snippet (one sample smoke test)
4. assumptions + edge cases still unresolved

## Guardrails

- Keep conversion code explicit; avoid hidden magic.
- Preserve user’s feature names unless they ask for renaming.
- If mesh conventions are uncertain, call that out and provide a safe fallback path.
- Prefer readable, debuggable code over over-optimized code.
- If required source variables are missing, stop and report the minimal set needed to proceed.

## Example prompts

### Example 1
Input: “I have one CSV per case with columns `rpm`, `tin`, `tout`, and a separate `pressure.npy`. Convert to PLAID Sample.”

Output behavior:

- produce mapping table
- map `rpm/tin/tout` as scalars
- map pressure as field
- provide constructor and one-sample check

### Example 2
Input: “Given my mesh nodes/connectivity arrays and solver outputs, create a `sample_constructor` returning a PLAID `Sample`.”

Output behavior:

- build tree/nodes/fields logic
- include field location assumptions
- include completeness checks
