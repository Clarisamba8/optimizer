"""Open/Closed registry — add new figure classes here only (OCP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures.ch03._fig43 import Fig43EquilibriumReturns
from theory.figures.ch03._fig45 import Fig45PickMatrix
from theory.figures.ch03._fig46 import Fig46BLPosterior
from theory.figures.ch03._fig48 import Fig48ConfidenceSensitivity
from theory.figures.ch03._fig49 import Fig49TrackRecord
from theory.figures.ch03._fig50 import Fig50FactorBL
from theory.figures.ch03._fig51 import Fig51EPReweight
from theory.figures.ch03._fig52 import Fig52EPViewTypes
from theory.figures.ch03._fig53 import Fig53OpinionPooling
from theory.figures.ch03._fig55 import Fig55SentimentMatrix


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
        access (none in ch03, kept for interface consistency).

    Returns
    -------
    list[FigureGenerator]
        Generators in the order they should be executed.
    """
    return [
        Fig43EquilibriumReturns(prices, output_dir),
        Fig45PickMatrix(prices, output_dir),
        Fig46BLPosterior(prices, output_dir),
        Fig48ConfidenceSensitivity(prices, output_dir),
        Fig49TrackRecord(prices, output_dir),
        Fig50FactorBL(prices, output_dir),
        Fig51EPReweight(prices, output_dir),
        Fig52EPViewTypes(prices, output_dir),
        Fig53OpinionPooling(prices, output_dir),
        Fig55SentimentMatrix(prices, output_dir),
    ]
