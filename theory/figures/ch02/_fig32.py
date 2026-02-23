"""Fig32EquilibriumVsHistorical — historical sample means vs equilibrium implied returns."""  # noqa: E501

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import EmpiricalMu, EquilibriumMu

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

# 3-year optimisation window consistent with ch01 / fig30
_WINDOW_DAYS = 756
# Cap the number of assets shown to keep the chart readable
_MAX_ASSETS = 30
_RISK_AVERSION = 2.5


class Fig32EquilibriumVsHistorical(FigureGenerator):
    """Grouped bar chart: historical sample means vs equilibrium implied returns.

    Both estimators are fitted on clean arithmetic returns over a 3-year window.
    Assets are sorted by descending last price so that larger-cap stocks appear
    on the left, making the chart easier to cross-reference against known names.
    Equal market-capitalisation weights are used as the implicit market portfolio
    inside :class:`EquilibriumMu` (the default when ``weights=None``).
    """

    @property
    def name(self) -> str:
        return "fig_32_equilibrium_vs_historical"

    def generate(self) -> None:
        prices = self._prices

        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        # Sort assets by last observed price (descending) and cap count
        last_prices = p_window[returns.columns].iloc[-1].sort_values(ascending=False)
        tickers = last_prices.index[:_MAX_ASSETS].tolist()
        returns = returns[tickers]

        n_assets = len(tickers)
        print(f"  Fig 32: {n_assets} assets x {len(returns)} days")

        emp = EmpiricalMu()
        emp.fit(returns)
        mu_hist = emp.mu_ * 252 * 100  # annualised %

        eq = EquilibriumMu(risk_aversion=_RISK_AVERSION)
        eq.fit(returns)
        mu_equil = eq.mu_ * 252 * 100  # annualised %

        short_names = [
            t.replace("_US_EQ", "").replace("_EQ", "") for t in tickers
        ]
        x = np.arange(n_assets)
        width = 0.40

        fig, ax = plt.subplots(figsize=(max(12, n_assets * 0.55), 5.5))

        ax.bar(x - width / 2, mu_hist,  width, color="#2196F3", alpha=0.80,
               label="Historical (EmpiricalMu)")
        ax.bar(x + width / 2, mu_equil, width, color="#FF9800", alpha=0.80,
               label=f"Equilibrium (EquilibriumMu, lambda={_RISK_AVERSION})")

        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Annualised Return (%)")
        ax.set_xlabel("Asset (sorted by last price, descending)")
        ax.set_title(
            "Historical Sample Means vs Equilibrium Implied Returns\n"
            f"(3-year window, {n_assets} assets, lambda={_RISK_AVERSION}, "
            "equal-weight market portfolio)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig)
