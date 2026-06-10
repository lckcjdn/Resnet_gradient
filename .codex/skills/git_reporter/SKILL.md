# Git Reporter Skill

## Purpose

This skill ensures code, docs, and experiment results are versioned.

## When to use

Use this skill:

- after project initialization;
- after implementing a major module;
- after completing an experiment group;
- before final delivery.

## Required commands

```bash
git status
git add .
git commit -m "<clear message>"
```

If remote upload is explicitly requested:

```bash
git push
```

If remote upload is deferred, record instructions in:

```text
docs/git_upload_instructions.md
```

## Commit message examples

```text
init: create project structure and docs
feat: implement plain and residual models
feat: add gradient monitor and logging harness
exp: add lambda shortcut ablation results
exp: add lesion study figures and tables
docs: update process conclusions
```

## Rules

1. Do not commit large datasets.
2. Do not commit unnecessary checkpoints unless explicitly required.
3. Commit final figures and CSV tables.
4. Use `.gitignore` to exclude raw data and heavy model files when needed.
