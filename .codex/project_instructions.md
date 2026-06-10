# Project Instructions

## Scope

Build a reproducible framework for testing ResNet gradient stability and short-path ensemble behavior. Keep changes small, documented, and verifiable.

## Operating Rules

1. Read `PROJECT_INIT.md` before task work.
2. Do not run full experiments unless explicitly requested.
3. Prefer existing modules and configs over introducing new abstractions.
4. Record experiment runs in `docs/experiment_log.md`.
5. Save generated CSV files under `results/tables/` and Markdown tables under `docs/tables/`.
6. Save figures under `results/figures/` and list them in `docs/expected_figures_and_tables.md`.
7. Avoid overclaiming. Use "supports", "suggests", and "is consistent with".
8. Keep checkpoints and raw data out of Git unless the user explicitly asks otherwise.

## Validation

Use the closest lightweight command first:

```bash
python -m harness.sanity_check
python -m compileall src harness scripts
```

Long training runs belong in later implementation phases.
