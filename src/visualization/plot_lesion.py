"""Lesion study plotting helpers."""

from __future__ import annotations


def plot_lesion_accuracy(frame, output_path: str, title: str = "") -> None:
    """Plot lesion accuracy against drop ratio."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in frame.groupby("drop_strategy"):
        ax.plot(group["drop_ratio"], group["test_accuracy"], marker="o", label=label)
    ax.set_xlabel("drop_ratio")
    ax.set_ylabel("test_accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
