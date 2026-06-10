# Result Auditor Skill

## Purpose

This skill checks whether figures and tables support the stated experimental hypothesis.

## When to use

Use this skill after generating:

- training curves;
- accuracy curves;
- gradient norm plots;
- gradient heatmaps;
- lambda ablation figures;
- lesion study figures;
- CSV summary tables.

## Audit questions

For each result, answer:

1. What hypothesis does this result test?
2. Which figure/table supports it?
3. Is the result consistent with the hypothesis?
4. Are there possible confounding factors?
5. Does the result need rerun with another seed?
6. Is the conclusion too strong?

## Required output

Append audit notes to:

```text
docs/process_conclusions.md
```

## Warning

Do not claim that ResNet completely eliminates vanishing gradients.
Use careful wording:

```text
supports
suggests
is consistent with
provides empirical evidence
```

Avoid overclaiming:

```text
proves completely
guarantees
always solves
```
