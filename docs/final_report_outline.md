# Final Report / PPT Outline

## Slide 1: Title

ResNet Gradient Stability and Short-Path Ensemble Behavior

## Slide 2: Motivation

Deep networks are powerful but hard to optimize.

## Slide 3: Degradation Problem

Plain deep networks may have higher training error.

## Slide 4: Residual Learning

`H(x) = F(x) + x`

## Slide 5: Identity Mapping

Identity shortcuts provide direct gradient propagation.

## Slide 6: Short-Path Ensemble

ResNet can be interpreted as many paths of different lengths.

## Slide 7: Experimental Design

PlainNet / ResNet / PreAct ResNet / Scaled Shortcut / Lesion Study

## Slide 8: Results - Loss and Accuracy

Use Fig. 1 and Fig. 2.

## Slide 9: Results - Gradient Stability

Use Fig. 3 and Fig. 4.

## Slide 10: Results - Shortcut Ablation

Use Fig. 5, Fig. 6, Fig. 7, and Table 3.

## Slide 11: Results - Lesion Study

Use Fig. 8, Fig. 9, Fig. 10, and Table 4.

## Slide 12: Conclusion

Use evidence-grounded wording:

- Identity shortcuts are associated with more stable gradient propagation in this setup.
- Lesion results can be used to evaluate short-path ensemble-like behavior.
- Results should be interpreted within the dataset, seed, and training budget limits.
