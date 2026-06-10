# Run Policy

1. Never overwrite previous experiment results.
2. Every run must have a unique run_id.
3. Every run must save config, metrics, logs, and gradient statistics.
4. Every completed run must update `docs/experiment_log.md`.
5. Every generated figure must be listed in `docs/expected_figures_and_tables.md`.
6. Every table must be saved as CSV and Markdown.
7. Any failed run must be recorded with error message and next action.
8. Before pushing to Git, run the sanity check.
