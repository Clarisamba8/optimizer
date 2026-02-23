"""Fig60RiskContributionPies — cap-weighted vs ERC risk contribution pie charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import (
    RiskBudgetingConfig,
    build_risk_budgeting,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8
_RNG_SEED = 42


class Fig60RiskContributionPies(FigureGenerator):
    """Two pie charts: cap-weighted RC vs ERC RC.

    Left panel shows concentrated risk contributions in a cap-weighted
    portfolio.  Right panel shows equal risk contributions under ERC.
    """

    @property
    def name(self) -> str:
        return "fig_60_risk_contributions"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        # Select assets with most data
        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()
        labels = [t.replace("_US_EQ", "").replace("_EQ", "") for t in top]

        cov = ret.cov().values

        # Cap-weighted proxy: use last price as market-cap proxy
        last_prices = prices[top].iloc[-1].values
        w_cap = last_prices / last_prices.sum()

        # ERC via optimizer
        erc = build_risk_budgeting(RiskBudgetingConfig.for_risk_parity())
        erc.fit(ret)
        w_erc = erc.weights_

        # Compute risk contributions: RC_i = w_i * (Sigma @ w)_i
        def risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            marginal = sigma @ w
            rc = w * marginal
            return rc / rc.sum()  # normalise to percentages

        rc_cap = risk_contributions(w_cap, cov)
        rc_erc = risk_contributions(w_erc, cov)

        print(
            f"  Fig 60: {_N_ASSETS} assets, "
            f"cap-wt top-3 RC = {np.sort(rc_cap)[-3:].sum():.1%}, "
            f"ERC max RC = {rc_erc.max():.1%}"
        )

        colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0",
                  "#E91E63", "#00BCD4", "#795548"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Cap-weighted
        _, _, autotexts1 = ax1.pie(
            rc_cap, labels=labels, autopct="%.0f%%", colors=colors[:_N_ASSETS],
            startangle=140, pctdistance=0.82,
        )
        for t in autotexts1:
            t.set_fontsize(8)
        ax1.set_title(
            "Cap-Weighted Risk Contributions\n(concentrated)",
            fontsize=10, fontweight="bold",
        )

        # Panel B: ERC
        _, _, autotexts2 = ax2.pie(
            rc_erc, labels=labels, autopct="%.0f%%", colors=colors[:_N_ASSETS],
            startangle=140, pctdistance=0.82,
        )
        for t in autotexts2:
            t.set_fontsize(8)
        ax2.set_title(
            "Equal Risk Contribution (ERC)\n(balanced)",
            fontsize=10, fontweight="bold",
        )

        fig.suptitle(
            "Risk Contribution: Cap-Weighted vs Equal Risk Contribution",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
