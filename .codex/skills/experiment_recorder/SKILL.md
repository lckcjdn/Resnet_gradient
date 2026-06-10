# Experiment Recorder Skill

## Purpose

This skill ensures every experiment is reproducible and documented.

## When to use

Use this skill after:

- creating a new experiment config;
- starting a training run;
- completing a training run;
- generating figures;
- generating tables;
- encountering an error.

## Required records

Every experiment must record:

```text
run_id
date_time
git_commit
config_path
model_name
dataset
optimizer
learning_rate
batch_size
epochs
seed
device
status
main_outputs
key_metrics
notes
```

## Documentation targets

Update:

```text
docs/experiment_log.md
docs/process_conclusions.md
results/runs/{run_id}/run_summary.md
```

## Rules

1. Do not overwrite old logs.
2. Append new records chronologically.
3. If a run fails, record the failure and error message.
4. If a result contradicts the expected hypothesis, record it honestly.
5. Save both raw CSV and human-readable Markdown summary.
