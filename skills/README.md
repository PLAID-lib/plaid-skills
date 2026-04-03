# PLAID conversion skills (Sample + storage focused)

This repository now includes a small bundle of skills dedicated to one workflow:

1. convert user-owned data into `plaid.Sample`
2. save converted samples with `plaid.storage.save_to_disk`
3. read back and validate roundtrip integrity with `plaid.storage.init_from_disk` + `Converter`

## Skill folders

- `skills/plaid-sample-converter/`
- `skills/plaid-storage-writer-pipeline/`
- `skills/plaid-storage-reader-roundtrip/`

## Shared resources

Common templates and helper scripts used across all three skills are provided in:

- `skills/plaid-conversion-shared/templates/`
- `skills/plaid-conversion-shared/scripts/`

## Notes

- The scope intentionally prioritizes `Sample` and `storage` modules.
- `evals/evals.json` files are included for each skill as initial trigger and behavior test prompts.