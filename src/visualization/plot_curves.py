"""Training curve plotting helpers."""

from __future__ import annotations


def plot_metric_curve(frame, x: str, y: str, hue: str, output_path: str, title: str = "") -> None:
    """Plot a metric curve from a pandas DataFrame."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in frame.groupby(hue):
        ax.plot(group[x], group[y], label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
