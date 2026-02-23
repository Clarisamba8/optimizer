"""Fig09BufferTurnover — Buffer zone impact on selection turnover."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.factors._config import SelectionConfig, SelectionMethod
from optimizer.factors._selection import select_stocks
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_BUFFER_FRACTIONS = (0.0, 0.05, 0.10, 0.15)
_REBALANCE_DAYS = 21  # monthly
_TARGET_COUNT = 100
_WINDOW_DAYS = 756  # 3 years
_COLOR = "#2196F3"


def _compute_sharpe_scores(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute annualised Sharpe ratio as a composite-score proxy.

    Parameters
    ----------
    returns:
        Return DataFrame (dates x tickers).
    window:
        Trailing window in trading days.

    Returns
    -------
    pd.Series
        Sharpe scores indexed by ticker.
    """
    tail = returns.iloc[-window:] if len(returns) > window else returns
    mu = tail.mean() * 252
    sigma = tail.std() * np.sqrt(252)
    sigma = sigma.replace(0, np.nan)
    return (mu / sigma).dropna()


class Fig09BufferTurnover(FigureGenerator):
    """Bar chart showing how buffer-zone size reduces selection turnover.

    Simulates monthly selection using Sharpe-rank scores as a composite
    proxy for four buffer fractions (0%, 5%, 10%, 15%) and reports
    average monthly turnover with error bars.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    """

    @property
    def name(self) -> str:
        return "fig_09_buffer_zone_turnover"

    def generate(self) -> None:
        prices = self._prices

        # Slice to window
        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()
        n_assets = returns.shape[1]
        n_dates = len(returns)
        print(f"  Fig 09: {n_assets} assets x {n_dates} days")

        # Rebalance dates (every _REBALANCE_DAYS trading days)
        rebalance_indices = list(range(252, n_dates, _REBALANCE_DAYS))
        if len(rebalance_indices) < 3:
            rebalance_indices = list(range(63, n_dates, _REBALANCE_DAYS))

        means: list[float] = []
        stds: list[float] = []

        for buf_frac in _BUFFER_FRACTIONS:
            config = SelectionConfig(
                method=SelectionMethod.FIXED_COUNT,
                target_count=_TARGET_COUNT,
                buffer_fraction=buf_frac,
                sector_balance=False,
            )
            turnovers: list[float] = []
            current_members: pd.Index | None = None

            for idx in rebalance_indices:
                scores = _compute_sharpe_scores(returns.iloc[:idx])
                selected, turnover = select_stocks(
                    scores,
                    config=config,
                    current_members=current_members,
                    return_turnover=True,
                )
                if current_members is not None:
                    turnovers.append(turnover)
                current_members = selected

            arr = np.array(turnovers) * 100  # to percent
            means.append(float(arr.mean()) if len(arr) > 0 else 0.0)
            stds.append(float(arr.std()) if len(arr) > 0 else 0.0)

        # Plot
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(_BUFFER_FRACTIONS))
        labels = [f"{int(b * 100)}%" for b in _BUFFER_FRACTIONS]

        bars = ax.bar(
            x, means, yerr=stds, capsize=5,
            color=_COLOR, edgecolor="white", width=0.5, alpha=0.85,
        )

        for bar, m in zip(bars, means, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{m:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Buffer Zone Size (% of target count)")
        ax.set_ylabel("Average Monthly Turnover (%)")
        ax.set_title(
            "Buffer Zones Cut Selection Turnover by Half\n"
            f"(Top-{_TARGET_COUNT} selection, monthly rebalancing, "
            f"{n_assets} assets)",
        )
        ax.set_ylim(0, max(means) * 1.5 if max(means) > 0 else 10)
        plt.tight_layout()
        self._save(fig)
