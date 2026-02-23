"""Open/Closed registry — add new figure classes here only (OCP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures.ch06._fig84 import Fig84WalkForward
from theory.figures.ch06._fig85 import Fig85RollingVsExpanding
from theory.figures.ch06._fig86 import Fig86CPCVFoldMatrix
from theory.figures.ch06._fig87 import Fig87CPCVSharpe
from theory.figures.ch06._fig89 import Fig89PerformanceRadar
from theory.figures.ch06._fig90 import Fig90GridSearch
from theory.figures.ch06._fig91 import Fig91Pipeline
from theory.figures.ch06._fig92 import Fig92RebalancingFrequency
from theory.figures.ch06._fig93 import Fig93TradeTriggers
from theory.figures.ch06._fig94 import Fig94GrossVsNet


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
        Fig84WalkForward(prices, output_dir),
        Fig85RollingVsExpanding(prices, output_dir),
        Fig86CPCVFoldMatrix(prices, output_dir),
        Fig87CPCVSharpe(prices, output_dir),
        Fig89PerformanceRadar(prices, output_dir),
        Fig90GridSearch(prices, output_dir),
        Fig91Pipeline(prices, output_dir),
        Fig92RebalancingFrequency(prices, output_dir),
        Fig93TradeTriggers(prices, output_dir),
        Fig94GrossVsNet(prices, output_dir),
    ]
