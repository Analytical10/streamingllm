from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from .logging_utils import ensure_dir


def plot_lines(
    x: Sequence[float],
    ys: Iterable[Sequence[float]],
    labels: Iterable[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(10, 5))
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
