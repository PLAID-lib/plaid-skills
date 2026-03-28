---
name: plaid-storage-writer-pipeline
description: Build and harden PLAID storage export pipelines using `plaid.storage.save_to_disk` from a user-provided `sample_constructor(id_) -> Sample`. Use this whenever a user asks to save converted samples, define train/test split ids, choose backend (`hf_datasets`, `zarr`, `cgns`), or generate a reproducible export script. Do not use this skill for raw-data-to-Sample mapping or post-export readback diagnostics.
---

# plaid-storage-writer-pipeline

Create reliable disk export pipelines after conversion to `Sample` is ready.

## Scope

Focus only on:

- `plaid.storage.save_to_disk`
- split identifiers (`ids` mapping)
- storage backend choice (`hf_datasets`, `zarr`, `cgns`)
- `infos` metadata and optional `pb_defs`

Use `push_to_hub` only if the user explicitly asks for Hub upload.

## Skill boundaries and handoffs

Use this skill for **writing datasets to storage** once `Sample` construction exists.

When requests move outside write-pipeline scope, hand off explicitly:

- For raw data mapping into `Sample` / `sample_constructor` design → use `plaid-sample-converter`.
- For loading datasets and roundtrip diagnostics with `init_from_disk` and `Converter` → use `plaid-storage-reader-roundtrip`.
- For high-level semantic conversion strategy (trajectory semantics, external time metadata, nodal vs cell mapping) → use `plaid-conversion`.

## Inputs to collect

1. where `sample_constructor` lives (or generate one if missing)
2. split ids (`train`, `test`, optional `val`)
3. output folder path
4. backend preference
5. multiprocessing constraints (`num_proc`)
6. overwrite policy

If split identifiers are malformed (e.g., generators, sets, mixed-type objects), stop and request a coercion plan to deterministic, sliceable sequences.

## Backend selection heuristics

- `hf_datasets`: default for ML workflows, split-centric organization, and convenient training ingestion.
- `zarr`: preferred for large chunked array workloads and high-throughput IO scenarios.
- `cgns`: use when users need CGNS-oriented interoperability and domain-specific compatibility.

When uncertain, ask the user's downstream consumption target first, then pick the backend.

## Pipeline workflow

1. **Validate prerequisites**
   - ensure constructor returns `Sample`
   - ensure each ids split is a sliceable sequence
   - if invalid, fail fast and show minimal coercion fix
2. **Draft storage config**
   - backend selection
   - `infos` minimum legal/data_production metadata
3. **Generate `save_to_disk(...)` script**
   - use `../plaid-conversion-shared/templates/save_to_disk_pipeline_template.py`
4. **Add a smoke test**
   - run on a tiny subset of ids first
   - then full write
5. **Document output layout and next steps**
   - where metadata and data files are stored

## Response protocol

1. Confirm constructor + split IDs before writing any script.
2. If IDs or output path policy is ambiguous, ask first rather than guessing.
3. Default to conservative execution settings (small subset, low `num_proc`) unless user requests aggressive parallelism.
4. Surface exact failure points and recovery steps for invalid IDs, overwrite conflicts, and constructor exceptions.

## Output contract

Always return:

1. ready-to-run script calling `save_to_disk`
2. explicit `ids` and backend configuration
3. short dry-run/smoke-run instructions
4. warnings about common pitfalls (non-sliceable ids, overwrite behavior, bad constructor)

## Guardrails

- Fail early if ids shape/type is invalid.
- Keep paths explicit and cross-platform where possible.
- Default to conservative settings (small `num_proc` unless user requests higher).
- Clearly state when user should use `overwrite=True`.
- Do not silently coerce IDs or mutate split semantics.
- Do not claim a write succeeded without an explicit smoke-run check.

## Example prompts

### Example 1
Input: “I already have `sample_constructor`, write me a script to save train/test splits to hf_datasets backend.”

Output behavior:

- create complete `save_to_disk` script
- include ids and metadata template
- include tiny-split smoke test instructions

### Example 2
Input: “Can you generate a zarr export pipeline with 16 workers for my converted samples?”

Output behavior:

- configure backend=zarr, num_proc=16
- include safeguards and overwrite policy
- provide expected output folder tree
