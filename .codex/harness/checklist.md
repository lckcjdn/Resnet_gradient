# Codex Execution Checklist

## Before coding

- [x] Read `PROJECT_INIT.md`.
- [x] Create project tree.
- [x] Create `.gitignore`.
- [x] Create docs directory.
- [x] Create `.codex` skills.

## Before running experiments

- [ ] Run `sanity_check.py`.
- [ ] Run one forward pass for each model.
- [ ] Run one mini training epoch.
- [ ] Confirm gradient monitor works.

## After each experiment

- [ ] Save `metrics.csv`.
- [ ] Save `gradient_stats.csv`.
- [ ] Save config copy.
- [ ] Save checkpoint.
- [ ] Update `experiment_log.md`.
- [ ] Update `process_conclusions.md`.

## Before final commit

- [ ] Generate all required figures.
- [ ] Generate all required tables.
- [ ] Update `docs/tables`.
- [ ] Update `final_report_outline.md`.
- [ ] Commit changes.
