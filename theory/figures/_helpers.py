"""Pure preprocessing helpers shared across all chapter figure generators.

This module also applies the project-wide matplotlib style at import time so
every figure benefits from consistent typography and spine settings without
each generator repeating the rcParams block.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._validation import DataValidator

# ---------------------------------------------------------------------------
# Global matplotlib style — applied once at import time
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# ---------------------------------------------------------------------------
# Pure return-computation helpers
# ---------------------------------------------------------------------------


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute arithmetic (simple) returns from adjusted close prices.

    Parameters
    ----------
    prices:
        Wide price DataFrame (date x ticker).

    Returns
    -------
    pd.DataFrame
        Simple percentage returns with the first row dropped.
    """
    return prices.pct_change().dropna(how="all")


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute logarithmic returns from adjusted close prices.

    Parameters
    ----------
    prices:
        Wide price DataFrame (date x ticker).

    Returns
    -------
    pd.DataFrame
        Log returns with the first row dropped.
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Apply DataValidator + OutlierTreater, then drop columns that become all-NaN.

    Removes:
    - Infinity / extreme prices caused by unit errors (e.g. GBX vs GBP)
    - Single-day blowups from corrupt price entries (e.g. Atos 10,130 EUR)
    - Rows/columns that are fully NaN after cleaning

    Parameters
    ----------
    returns:
        Raw arithmetic return DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned return DataFrame with columns that lost >10% of observations
        removed entirely.
    """
    validator = DataValidator(max_abs_return=10.0)   # flag |r| > 1000%
    validator.fit(returns)
    out = validator.transform(returns)

    treater = OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)
    treater.fit(out)
    out = treater.transform(out)

    # Drop columns that lost too many observations (>10% removed)
    keep = out.notna().mean() >= 0.90
    return out.loc[:, keep]
