"""Open/Closed registry — add new figure classes here only (OCP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures.ch05._fig72 import Fig72EfficientFrontier
from theory.figures.ch05._fig74 import Fig74Regularization
from theory.figures.ch05._fig75 import Fig75ConstrainedFrontier
from theory.figures.ch05._fig76 import Fig76RobustEllipse
from theory.figures.ch05._fig78 import Fig78DRCVaR
from theory.figures.ch05._fig79 import Fig79VineCopula
from theory.figures.ch05._fig80 import Fig80StressFan
from theory.figures.ch05._fig81 import Fig81BenchmarkTracking
from theory.figures.ch05._fig82 import Fig82StrategyComparison
from theory.figures.ch05._fig83 import Fig83StackingSchematic


def build_generators(
    prices: pd.DataFrame,
    output_dir: Path,
    db_url: str,
) -> list[FigureGenerator]:
    """Construct and return all figure generators in render order.

    Parameters
    ----------
    prices:
        Wide price DataFrame loaded by :class:`~theory.figures._loader.PriceLoader`.
    output_dir:
        Directory where PNG files are written.
    db_url:
        Database connection string (unused in this chapter but kept for
        interface consistency).

    Returns
    -------
    list[FigureGenerator]
        Generators in the order they should be executed.
    """
    return [
        Fig72EfficientFrontier(prices, output_dir),
        Fig74Regularization(prices, output_dir),
        Fig75ConstrainedFrontier(prices, output_dir),
        Fig76RobustEllipse(prices, output_dir),
        Fig78DRCVaR(prices, output_dir),
        Fig79VineCopula(prices, output_dir),
        Fig80StressFan(prices, output_dir),
        Fig81BenchmarkTracking(prices, output_dir),
        Fig82StrategyComparison(prices, output_dir),
        Fig83StackingSchematic(prices, output_dir),
    ]
