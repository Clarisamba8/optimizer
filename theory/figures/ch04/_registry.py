"""Open/Closed registry — add new figure classes here only (OCP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures.ch04._fig56 import Fig56AnnotatedRiskMeasures
from theory.figures.ch04._fig57 import Fig57CVaRVsVaR
from theory.figures.ch04._fig59 import Fig59RiskRadar
from theory.figures.ch04._fig60 import Fig60RiskContributionPies
from theory.figures.ch04._fig62 import Fig62ERCComparison
from theory.figures.ch04._fig63 import Fig63DiversificationScatter
from theory.figures.ch04._fig64 import Fig64DistanceHeatmaps
from theory.figures.ch04._fig65 import Fig65WardDendrogram
from theory.figures.ch04._fig66 import Fig66HRPWeights
from theory.figures.ch04._fig68 import Fig68HERCvsHRP
from theory.figures.ch04._fig69 import Fig69NCOSchematic
from theory.figures.ch04._fig70 import Fig70RegimeRisk
from theory.figures.ch04._fig71 import Fig71RegimeBudgetRotation


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
        access (e.g. sector colors for dendrograms).

    Returns
    -------
    list[FigureGenerator]
        Generators in the order they should be executed.
    """
    return [
        Fig56AnnotatedRiskMeasures(prices, output_dir),
        Fig57CVaRVsVaR(prices, output_dir),
        Fig59RiskRadar(prices, output_dir),
        Fig60RiskContributionPies(prices, output_dir),
        Fig62ERCComparison(prices, output_dir),
        Fig63DiversificationScatter(prices, output_dir),
        Fig64DistanceHeatmaps(prices, output_dir),
        Fig65WardDendrogram(prices, output_dir, db_url=db_url),
        Fig66HRPWeights(prices, output_dir),
        Fig68HERCvsHRP(prices, output_dir, db_url=db_url),
        Fig69NCOSchematic(prices, output_dir),
        Fig70RegimeRisk(prices, output_dir),
        Fig71RegimeBudgetRotation(prices, output_dir),
    ]
