"""Gradient plotting helpers."""

from __future__ import annotations


def plot_layerwise_gradient(frame, output_path: str, title: str = "") -> None:
    """Plot layer-wise log gradient norms."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in frame.groupby("model"):
        ax.plot(group["layer_index"], group["log10_grad_norm"], marker="o", label=label)
    ax.set_xlabel("layer_index")
    ax.set_ylabel("log10_grad_norm")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
