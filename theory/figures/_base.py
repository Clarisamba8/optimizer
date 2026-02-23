"""Abstract base class for all figure generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class FigureGenerator(ABC):
    """Abstract base for all figure generators.

    Each concrete subclass encapsulates a single figure's logic (SRP).
    All subclasses are substitutable (LSP) and depend only on a clean
    ``pd.DataFrame`` of prices and a ``Path`` output directory (DIP).

    Parameters
    ----------
    prices:
        Wide price DataFrame (date x ticker) already loaded from the DB.
    output_dir:
        Directory where the generated PNG is saved.
    """

    def __init__(self, prices: pd.DataFrame, output_dir: Path) -> None:
        self._prices = prices
        self._output_dir = output_dir

    @property
    @abstractmethod
    def name(self) -> str:
        """Figure filename stem, e.g. ``"fig_17_arithmetic_vs_log"``."""

    @abstractmethod
    def generate(self) -> None:
        """Render and save the figure to ``output_dir / name + '.png'``."""

    def _save(self, fig: plt.Figure) -> None:
        """Save the figure to disk and close it.

        Parameters
        ----------
        fig:
            Matplotlib figure to save.
        """
        path = self._output_dir / f"{self.name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {self.name}.png")
