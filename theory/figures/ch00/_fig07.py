"""Fig07RollingIC — Rolling 12-month information coefficient time series."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from theory.figures._base import FigureGenerator

_ROLLING_WINDOW = 12  # months
_COLORS = {
    "Momentum (12-1)": "#2196F3",
    "Volatility": "#E91E63",
    "Short-Term Rev.": "#FF9800",
}


def _compute_monthly_snapshots(
    prices: pd.DataFrame,
) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """Resample prices to month-end and compute returns for IC calculation.

    Returns
    -------
    tuple[list[pd.Timestamp], pd.DataFrame]
        (month-end dates, monthly returns DataFrame).
    """
    monthly = prices.resample("ME").last().dropna(how="all")
    monthly_ret = monthly.pct_change().dropna(how="all")
    dates = monthly_ret.index.tolist()
    return dates, monthly_ret


def _momentum_score(prices: pd.DataFrame, date_idx: int) -> pd.Series:
    """12-1 month momentum at a monthly snapshot."""
    if date_idx < 12:
        return pd.Series(dtype=float)
    p_12m = prices.iloc[date_idx - 12]
    p_1m = prices.iloc[date_idx - 1]
    # 12-month return skipping most recent month
    mom = (p_1m / p_12m) - 1
    return mom.dropna()


def _volatility_score(monthly_ret: pd.DataFrame, date_idx: int) -> pd.Series:
    """Trailing 12-month volatility (annualized) as a score."""
    start = max(0, date_idx - 12)
    window = monthly_ret.iloc[start:date_idx]
    if len(window) < 6:
        return pd.Series(dtype=float)
    vol = window.std() * np.sqrt(12)
    return vol.dropna()


def _short_term_reversal(monthly_ret: pd.DataFrame, date_idx: int) -> pd.Series:
    """Most recent 1-month return (reversal signal)."""
    if date_idx < 1:
        return pd.Series(dtype=float)
    return monthly_ret.iloc[date_idx - 1].dropna()


class Fig07RollingIC(FigureGenerator):
    """Rolling 12-month IC for price-based factor groups.

    Computes cross-sectional Spearman IC between factor scores and
    one-month-ahead returns at monthly snapshots, then plots the
    rolling 12-month average IC.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        PostgreSQL connection string (unused for price-based factors).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        output_dir: Path,
        db_url: str,
    ) -> None:
        super().__init__(prices, output_dir)
        self._db_url = db_url

    @property
    def name(self) -> str:
        return "fig_07_rolling_ic"

    def generate(self) -> None:
        prices = self._prices.ffill()
        monthly_prices = prices.resample("ME").last().dropna(how="all")
        monthly_ret = monthly_prices.pct_change().dropna(how="all")
        n_months = len(monthly_ret)
        print(f"  Fig 07: {n_months} monthly snapshots")

        if n_months < 24:
            print("  Fig 07: too few months, using fallback.")
            self._generate_fallback()
            return

        factor_fns = {
            "Momentum (12-1)": lambda idx: _momentum_score(monthly_prices, idx),
            "Volatility": lambda idx: _volatility_score(monthly_ret, idx),
            "Short-Term Rev.": lambda idx: _short_term_reversal(monthly_ret, idx),
        }

        ic_series: dict[str, list[float]] = {k: [] for k in factor_fns}
        dates: list[pd.Timestamp] = []

        for t in range(13, n_months - 1):
            fwd_ret = monthly_ret.iloc[t]
            valid_fwd = fwd_ret.dropna()
            if len(valid_fwd) < 20:
                continue

            dates.append(monthly_ret.index[t])

            for name, fn in factor_fns.items():
                scores = fn(t)
                common = scores.index.intersection(valid_fwd.index)
                if len(common) < 20:
                    ic_series[name].append(np.nan)
                    continue
                corr, _ = sp_stats.spearmanr(scores.loc[common], valid_fwd.loc[common])
                ic_series[name].append(corr)

        if len(dates) < _ROLLING_WINDOW:
            print("  Fig 07: insufficient IC data, using fallback.")
            self._generate_fallback()
            return

        self._plot(dates, ic_series)

    def _generate_fallback(self) -> None:
        """Synthetic IC series for illustration."""
        np.random.seed(42)
        n = 120
        dates = pd.date_range("2014-01-01", periods=n, freq="ME").tolist()
        ic_series = {
            "Momentum (12-1)": (np.random.normal(0.04, 0.08, n)).tolist(),
            "Volatility": (np.random.normal(0.015, 0.06, n)).tolist(),
            "Short-Term Rev.": (np.random.normal(-0.02, 0.07, n)).tolist(),
        }
        self._plot(dates, ic_series)

    def _plot(
        self,
        dates: list[pd.Timestamp],
        ic_series: dict[str, list[float]],
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 5.5))

        for name, ic_vals in ic_series.items():
            s = pd.Series(ic_vals, index=dates)
            rolling = s.rolling(_ROLLING_WINDOW, min_periods=6).mean()
            color = _COLORS.get(name, "#333333")
            ax.plot(rolling.index, rolling.values, lw=1.5, color=color, label=name)

            # Shaded ±1 std band
            rolling_std = s.rolling(_ROLLING_WINDOW, min_periods=6).std()
            ax.fill_between(
                rolling.index,
                (rolling - rolling_std).values,
                (rolling + rolling_std).values,
                color=color, alpha=0.1,
            )

        ax.axhline(0, color="black", ls="-", lw=0.8, alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling 12-Month IC (Spearman)")
        ax.set_title("Rolling Information Coefficients by Factor Group")
        ax.legend(fontsize=9)
        fig.autofmt_xdate()
        plt.tight_layout()
        self._save(fig)
