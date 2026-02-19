# Contributing Conversion Skills

This repository stores **dataset-specific conversion knowledge** for PLAID.

Please optimize for scientific correctness and explicit semantics, not for generic abstraction.

---

## Contribution checklist

When adding a new conversion example:

- [ ] Add one script per dataset under `skills/examples/conversions/`.
- [ ] Include a module docstring with: dataset name, purpose, and key semantics.
- [ ] Keep all dataset assumptions explicit (paths, metadata, split logic, field locations).
- [ ] Define `infos` metadata (`legal`, `data_production`) with data source/license.
- [ ] Define a `ProblemDefinition` with clear in/out/constant feature identifiers.
- [ ] Preserve semantic choices explicitly (static vs temporal, trajectory grouping, node vs cell).
- [ ] Keep external dependencies local to the script; do not expand repo-wide deps for one dataset.
- [ ] Use placeholder assertions for required local paths/repo IDs.
- [ ] Validate on a small subset before full conversion.
- [ ] If you introduce a recurring semantic pattern, add/update a file in `skills/patterns/`.

---

## Do / Don't

### Do

- Prefer explicit conversion logic over hidden abstractions.
- Add comments where domain semantics are non-obvious.
- Keep one `Sample` per scientifically meaningful unit (e.g., one trajectory).
- Document any intentional subsampling or preprocessing.

### Don't

- Don’t turn scripts into generic frameworks.
- Don’t invent one-off PLAID APIs for a specific dataset.
- Don’t refactor existing scripts just to enforce uniform style.
- Don’t silently change scientific meaning while simplifying code.

---

## Starter scaffold

Use [`skills/examples/conversions/_template.py`](./skills/examples/conversions/_template.py) as a starting point.
