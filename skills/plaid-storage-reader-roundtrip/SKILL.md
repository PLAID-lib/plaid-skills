---
name: plaid-storage-reader-roundtrip
description: Load PLAID datasets from disk (or Hub when requested) and validate conversion roundtrip using `plaid.storage.init_from_disk` and `Converter` methods (`to_plaid`, `to_dict`, `plaid_to_dict`). Use this whenever a user asks to inspect already-saved data, debug missing features after export, verify fidelity after `save_to_disk`, or compare stored content to expected schema. Do not use this skill for raw data mapping into `Sample` or authoring write pipelines.
---

# plaid-storage-reader-roundtrip

Validate that saved PLAID data can be read and reconstructed correctly.

## Scope

Focus on:

- `plaid.storage.init_from_disk`
- optional `download_from_hub` / `init_streaming_from_hub` (only on user request)
- `Converter` usage for reconstruction and dict conversion
- practical roundtrip checks (presence, shape, consistency)

## Skill boundaries and handoffs

Use this skill for **post-export inspection and roundtrip diagnostics**.

When requests are outside this scope, hand off explicitly:

- For raw dataset-to-`Sample` mapping and constructor authoring → use `plaid-sample-converter`.
- For `save_to_disk` pipeline setup, split IDs, and backend export configuration → use `plaid-storage-writer-pipeline`.
- For high-level semantic conversion design (trajectory semantics, external time metadata, nodal vs cell mapping) → use `plaid-conversion`.

## Inputs to collect

1. dataset folder path (or `repo_id` if Hub)
2. target split(s) to inspect
3. critical features/scalars expected by user
4. expected sample count and rough shapes
5. whether comparison should be strict or smoke-level

If local path/split information is missing, ask for the minimal required inputs first (`local_dir`, split name, and one expected feature).

## Roundtrip workflow

1. **Initialize reader**
   - call `init_from_disk(local_dir)`
   - pick split dataset + converter
2. **Reconstruct sample(s)**
   - use `converter.to_plaid(dataset, idx)`
3. **Validate content**
   - check scalar names/features are present
   - check non-empty arrays and key shapes
4. **Compare representations**
   - where relevant, compare `to_dict` / `plaid_to_dict` outputs
5. **Report failures with fixes**
   - missing feature path
   - incorrect location/naming
   - split mismatch or wrong ids

## Triage matrix

- **Read/init errors** (`init_from_disk` fails, path not found, split absent): classify as storage/layout issue.
- **Reconstruction errors** (`to_plaid` fails): classify as converter/schema compatibility issue.
- **Presence/shape mismatches** (feature missing or malformed): classify as conversion mapping issue.
- **Representation divergence** (`to_dict` vs `plaid_to_dict` inconsistency): classify as roundtrip fidelity issue.

Report the class of failure explicitly before proposing fixes.

## Response protocol

1. Start with a tiny subset and deterministic checks.
2. Fail fast on missing required paths/features; do not continue with partial assumptions.
3. Distinguish storage read failures from conversion semantic failures.
4. Return actionable next steps tied to the first failing check.

Use `../plaid-conversion-shared/templates/roundtrip_validation_template.py` as base.

## Output contract

Always return:

1. runnable inspection script
2. explicit checks and pass/fail criteria
3. short interpretation of expected outputs
4. targeted troubleshooting steps if checks fail

## Guardrails

- Do not silently ignore missing required features.
- Distinguish backend read issues from conversion mapping issues.
- Keep checks deterministic and fast on a small subset first.

## Example prompts

### Example 1
Input: “I saved with `save_to_disk`; now show me how to load and verify pressure field exists in train split.”

Output behavior:

- initialize reader
- inspect a few indices
- assert pressure field presence + shape sanity

### Example 2
Input: “My restored samples look wrong. Build a roundtrip debug script using Converter methods.”

Output behavior:

- reconstruct samples with `to_plaid`
- compare dictionary representations
- print actionable mismatch diagnostics
