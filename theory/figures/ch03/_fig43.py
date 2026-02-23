"""Fig43EquilibriumReturns — market-implied equilibrium returns by sector."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import LedoitWolf

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 756
_MAX_ASSETS = 10
_RISK_AVERSION = 2.5


class Fig43EquilibriumReturns(FigureGenerator):
    """Horizontal bar chart of equilibrium returns with market-cap weights.

    Computes Pi = delta * Sigma * w_mkt for a set of assets sorted by
    implied return, with a secondary axis showing market-capitalisation
    weights (approximated as equal weights, consistent with EquilibriumMu
    defaults).
    """

    @property
    def name(self) -> str:
        return "fig_43_equilibrium_returns"

    def generate(self) -> None:
        prices = self._prices
        p_window = (
            prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        )
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        # Select top assets by last price as proxy for market cap
        last_prices = p_window[returns.columns].iloc[-1].sort_values(ascending=False)
        tickers = last_prices.index[:_MAX_ASSETS].tolist()
        returns = returns[tickers]

        n_assets = len(tickers)
        print(f"  Fig 43: {n_assets} assets x {len(returns)} days")

        # Fit covariance
        lw = LedoitWolf()
        lw.fit(returns)
        sigma = lw.covariance_

        # Equal market-cap weights (EquilibriumMu default)
        w_mkt = np.ones(n_assets) / n_assets

        # Pi = delta * Sigma * w_mkt
        pi = _RISK_AVERSION * sigma @ w_mkt
        pi_annual = pi * 252 * 100  # annualised %

        # Sort by implied return
        sort_idx = np.argsort(pi_annual)
        pi_sorted = pi_annual[sort_idx]
        w_sorted = w_mkt[sort_idx] * 100  # %
        names_sorted = [
            tickers[i].replace("_US_EQ", "").replace("_EQ", "")
            for i in sort_idx
        ]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(n_assets)

        # Horizontal bars for Pi
        ax1.barh(
            y_pos, pi_sorted, height=0.6, color="#2196F3", alpha=0.85,
            label=r"$\Pi_i$ (equilibrium return)",
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names_sorted, fontsize=9)
        ax1.set_xlabel("Annualised Equilibrium Return (%)")
        ax1.set_title(
            "Market-Implied Equilibrium Returns by Sector\n"
            rf"$\Pi = \delta \Sigma w_{{mkt}}$, $\delta$={_RISK_AVERSION}, "
            f"LedoitWolf covariance, {n_assets} assets",
            fontsize=11, fontweight="bold",
        )

        # Secondary axis for market-cap weights
        ax2 = ax1.twiny()
        ax2.plot(
            w_sorted, y_pos, "D", color="#FF9800", markersize=7, alpha=0.9,
            label="Market weight (%)",
        )
        ax2.set_xlabel("Market Weight (%)", color="#FF9800")
        ax2.tick_params(axis="x", colors="#FF9800")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

        plt.tight_layout()
        self._save(fig)
