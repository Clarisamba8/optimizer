"""Open/Closed registry — add new figure classes here only (OCP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures.ch00._fig01 import Fig01HysteresisTurnover
from theory.figures.ch00._fig02 import Fig02CrossFactorCorrelation
from theory.figures.ch00._fig04 import Fig04RankNormalHistograms
from theory.figures.ch00._fig05 import Fig05SectorNeutralization
from theory.figures.ch00._fig06 import Fig06AlphaScoreDistribution
from theory.figures.ch00._fig07 import Fig07RollingIC
from theory.figures.ch00._fig08 import Fig08VIFBarChart
from theory.figures.ch00._fig09 import Fig09BufferTurnover
from theory.figures.ch00._fig10 import Fig10NetAlpha
from theory.figures.ch00._fig11 import Fig11SectorBalance
from theory.figures.ch00._fig12 import Fig12RegimeTiltHeatmap
from theory.figures.ch00._fig13 import Fig13TimeVaryingWeights
from theory.figures.ch00._fig14 import Fig14QuintileSpreads
from theory.figures.ch00._fig15 import Fig15ISvsOOSIC


def build_generators(
    prices: pd.DataFrame,
    output_dir: Path,
    db_url: str,
) -> list[FigureGenerator]:
    """Construct and return all Ch00 figure generators in render order.

    Parameters
    ----------
    prices:
        Wide price DataFrame loaded by :class:`~theory.figures._loader.PriceLoader`.
    output_dir:
        Directory where PNG files are written.
    db_url:
        Database connection string forwarded to figures that need raw DB
        access.

    Returns
    -------
    list[FigureGenerator]
        Generators in the order they should be executed.
    """
    return [
        Fig01HysteresisTurnover(prices, output_dir, db_url),
        Fig02CrossFactorCorrelation(prices, output_dir, db_url),
        Fig04RankNormalHistograms(prices, output_dir, db_url),
        Fig05SectorNeutralization(prices, output_dir, db_url),
        Fig06AlphaScoreDistribution(prices, output_dir, db_url),
        Fig07RollingIC(prices, output_dir, db_url),
        Fig08VIFBarChart(prices, output_dir, db_url),
        Fig09BufferTurnover(prices, output_dir),
        Fig10NetAlpha(prices, output_dir),
        Fig11SectorBalance(prices, output_dir, db_url),
        Fig12RegimeTiltHeatmap(prices, output_dir),
        Fig13TimeVaryingWeights(prices, output_dir, db_url),
        Fig14QuintileSpreads(prices, output_dir, db_url),
        Fig15ISvsOOSIC(prices, output_dir, db_url),
    ]
