"""Fig62ERCComparison — grouped bar chart of portfolio weights under 5 strategies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import (
    MaxDiversificationConfig,
    MeanRiskConfig,
    RiskBudgetingConfig,
    build_equal_weighted,
    build_inverse_volatility,
    build_max_diversification,
    build_mean_risk,
    build_risk_budgeting,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 6


class Fig62ERCComparison(FigureGenerator):
    """Grouped bars: 5 strategies x N assets showing weight comparison.

    EW, InverseVol, MinVar, ERC, MaxDiv — demonstrates that ERC
    lies between equal-weight and minimum variance.
    """

    @property
    def name(self) -> str:
        return "fig_62_erc_comparison"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()
        labels = [t.replace("_US_EQ", "").replace("_EQ", "") for t in top]

        # Build 5 strategies
        strategies = {}

        ew = build_equal_weighted()
        ew.fit(ret)
        strategies["Equal Weight"] = ew.weights_

        iv = build_inverse_volatility()
        iv.fit(ret)
        strategies["Inverse Vol"] = iv.weights_

        mv = build_mean_risk(MeanRiskConfig.for_min_variance())
        mv.fit(ret)
        strategies["Min Variance"] = mv.weights_

        erc = build_risk_budgeting(RiskBudgetingConfig.for_risk_parity())
        erc.fit(ret)
        strategies["ERC"] = erc.weights_

        md = build_max_diversification(MaxDiversificationConfig())
        md.fit(ret)
        strategies["Max Diversification"] = md.weights_

        colors = ["#9E9E9E", "#FF9800", "#2196F3", "#4CAF50", "#E91E63"]
        strat_names = list(strategies.keys())

        x = np.arange(len(labels))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (name, w) in enumerate(strategies.items()):
            offset = (i - 2) * width
            ax.bar(x + offset, w, width, label=name, color=colors[i], alpha=0.85)

        ax.set_xlabel("Asset")
        ax.set_ylabel("Portfolio Weight")
        ax.set_title(
            "ERC Weights: Between Equal-Weight and Minimum Variance",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

        print(f"  Fig 62: {len(strat_names)} strategies x {_N_ASSETS} assets")

        plt.tight_layout()
        self._save(fig)
