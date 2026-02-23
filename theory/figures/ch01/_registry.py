"""Open/Closed registry — add new figure classes here only (OCP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures.ch01._fig17 import Fig17ArithmeticVsLog
from theory.figures.ch01._fig19 import Fig19ScalingComparison
from theory.figures.ch01._fig21 import Fig21OutlierGroups
from theory.figures.ch01._fig23 import Fig23DataQuality
from theory.figures.ch01._fig24 import Fig24CorrelationDistribution
from theory.figures.ch01._fig25 import Fig25ParetoFront
from theory.figures.ch01._fig27 import Fig27PreselectionFunnel


def build_generators(
    prices: pd.DataFrame,
    output_dir: Path,
    db_url: str,
) -> list[FigureGenerator]:
    """Construct and return all figure generators in render order.

    To add a new figure, create a :class:`FigureGenerator` subclass and
    append it to the list below.  No other code needs to change (OCP).

    Parameters
    ----------
    prices:
        Wide price DataFrame loaded by :class:`~theory.figures._loader.PriceLoader`.
    output_dir:
        Directory where PNG files are written.
    db_url:
        Database connection string forwarded to figures that need raw DB
        access (currently only :class:`Fig27PreselectionFunnel`).

    Returns
    -------
    list[FigureGenerator]
        Generators in the order they should be executed.
    """
    return [
        Fig17ArithmeticVsLog(prices, output_dir),
        Fig19ScalingComparison(prices, output_dir),
        Fig21OutlierGroups(prices, output_dir),
        Fig23DataQuality(prices, output_dir),
        Fig24CorrelationDistribution(prices, output_dir),
        Fig25ParetoFront(prices, output_dir),
        Fig27PreselectionFunnel(prices, output_dir, db_url),
    ]
