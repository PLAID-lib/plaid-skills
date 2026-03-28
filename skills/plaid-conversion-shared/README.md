# Shared resources for PLAID conversion skills

This folder contains reusable building blocks referenced by all conversion skills.

## Templates

- `templates/mapping_table_template.md`: structured raw-to-PLAID mapping worksheet
- `templates/sample_constructor_template.py`: starter `sample_constructor(id_) -> Sample`
- `templates/save_to_disk_pipeline_template.py`: starter dataset export script
- `templates/roundtrip_validation_template.py`: starter read-back validation script

## Scripts

- `scripts/check_sample_requirements.py`: helper to assert required scalars and feature paths for one sample
