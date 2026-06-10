"""Minimal tests for model construction and lesion utilities."""

from __future__ import annotations

import unittest


class CoreTest(unittest.TestCase):
    def test_model_forward_shapes(self) -> None:
        import torch

        from src.models import build_model

        inputs = torch.randn(2, 3, 32, 32)
        for name in ["PlainNet", "ResNetV1", "PreActResNet", "ScaledShortcutResNet"]:
            model = build_model(name, depth=20, shortcut_lambda=1.0)
            outputs = model(inputs)
            self.assertEqual(tuple(outputs.shape), (2, 10))

    def test_lesion_mask_selection(self) -> None:
        from src.analysis.lesion import make_block_masks, select_dropped_blocks

        dropped = select_dropped_blocks(10, 0.3, "early_drop", seed=0)
        self.assertEqual(dropped, [0, 1, 2])
        masks = make_block_masks(5, [1, 3])
        self.assertEqual(masks, [1.0, 0.0, 1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
