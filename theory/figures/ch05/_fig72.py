"""Fig72EfficientFrontier — efficient frontier with key optimization objectives."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import (
    MeanRiskConfig,
    build_mean_risk,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 10


class Fig72EfficientFrontier(FigureGenerator):
    """Efficient frontier with min-var, max-Sharpe, max-utility + CML.

    Uses real DB prices to build the frontier and mark key portfolios.
    """

    @property
    def name(self) -> str:
        return "fig_72_efficient_frontier"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        # Build efficient frontier via fit_predict -> Population
        ef_cfg = MeanRiskConfig.for_efficient_frontier(size=30)
        ef = build_mean_risk(ef_cfg)
        population = ef.fit_predict(ret)

        # Extract frontier points
        frontier_returns = np.array([p.mean for p in population])
        frontier_risks = np.array(
            [np.sqrt(p.variance) for p in population]
        )

        # Key portfolios
        mv = build_mean_risk(MeanRiskConfig.for_min_variance())
        mv.fit(ret)
        mv_ret = mv.predict(ret).mean
        mv_risk = np.sqrt(mv.predict(ret).variance)

        ms = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        ms.fit(ret)
        ms_ret = ms.predict(ret).mean
        ms_risk = np.sqrt(ms.predict(ret).variance)

        # Max utility for different lambdas
        utility_points = {}
        for lam in [1, 5, 10]:
            mu = build_mean_risk(
                MeanRiskConfig.for_max_utility(risk_aversion=lam)
            )
            mu.fit(ret)
            p = mu.predict(ret)
            utility_points[lam] = (np.sqrt(p.variance), p.mean)

        fig, ax = plt.subplots(figsize=(12, 7))

        ann_factor_ret = 252
        ann_factor_risk = np.sqrt(252)

        fr_ann = frontier_returns * ann_factor_ret * 100
        fk_ann = frontier_risks * ann_factor_risk * 100

        ax.plot(
            fk_ann, fr_ann, "b-", linewidth=2.5,
            label="Efficient Frontier",
        )

        ax.plot(
            mv_risk * ann_factor_risk * 100,
            mv_ret * ann_factor_ret * 100,
            "o", color="#4CAF50", markersize=12, zorder=5,
            label="Min Variance",
        )

        ax.plot(
            ms_risk * ann_factor_risk * 100,
            ms_ret * ann_factor_ret * 100,
            "D", color="#E91E63", markersize=12, zorder=5,
            label="Max Sharpe (Tangency)",
        )

        # CML ray
        ms_x = ms_risk * ann_factor_risk * 100
        ms_y = ms_ret * ann_factor_ret * 100
        cml_x = np.linspace(0, fk_ann.max() * 1.1, 100)
        sharpe = ms_y / ms_x if ms_x > 0 else 0
        cml_y = sharpe * cml_x
        ax.plot(
            cml_x, cml_y, "--", color="#E91E63", linewidth=1.5,
            alpha=0.6, label="Capital Market Line",
        )

        utility_colors = ["#FF9800", "#9C27B0", "#795548"]
        for (lam, (risk, ret_)), color in zip(
            utility_points.items(), utility_colors, strict=True
        ):
            ax.plot(
                risk * ann_factor_risk * 100,
                ret_ * ann_factor_ret * 100,
                "s", color=color, markersize=10, zorder=5,
                label=rf"Max Utility ($\lambda={lam}$)",
            )

        ax.set_xlabel("Annualized Risk (Std Dev, %)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title(
            "Efficient Frontier with Key Optimization Objectives",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        print(
            f"  Fig 72: frontier with {len(population)} points, "
            f"{_N_ASSETS} assets"
        )

        plt.tight_layout()
        self._save(fig)
