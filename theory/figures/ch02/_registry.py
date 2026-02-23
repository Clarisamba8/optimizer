"""Open/Closed registry — add new figure classes here only (OCP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures.ch02._fig28 import Fig28SampleMeanInstability
from theory.figures.ch02._fig30 import Fig30ShrinkageScatter
from theory.figures.ch02._fig32 import Fig32EquilibriumVsHistorical
from theory.figures.ch02._fig33 import Fig33LWEigenvalues
from theory.figures.ch02._fig34 import Fig34MarchenkoPastur
from theory.figures.ch02._fig36 import Fig36Detoning
from theory.figures.ch02._fig37 import Fig37GerberVsPearson
from theory.figures.ch02._fig38 import Fig38FactorModelParams
from theory.figures.ch02._fig39 import Fig39HMMRegimes
from theory.figures.ch02._fig40 import Fig40HMMBlended
from theory.figures.ch02._fig42 import Fig42DMMvsHMM


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
        access (currently only :class:`Fig37GerberVsPearson`).

    Returns
    -------
    list[FigureGenerator]
        Generators in the order they should be executed.
    """
    return [
        Fig28SampleMeanInstability(prices, output_dir),
        Fig30ShrinkageScatter(prices, output_dir),
        Fig32EquilibriumVsHistorical(prices, output_dir),
        Fig33LWEigenvalues(prices, output_dir),
        Fig34MarchenkoPastur(prices, output_dir),
        Fig36Detoning(prices, output_dir),
        Fig37GerberVsPearson(prices, output_dir, db_url),
        Fig38FactorModelParams(prices, output_dir),
        Fig39HMMRegimes(prices, output_dir),
        Fig40HMMBlended(prices, output_dir),
        Fig42DMMvsHMM(prices, output_dir),
    ]
