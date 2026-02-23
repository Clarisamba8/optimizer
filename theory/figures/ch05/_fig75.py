"""Fig75ConstrainedFrontier — efficient frontiers under different constraint sets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 10


class Fig75ConstrainedFrontier(FigureGenerator):
    """Overlay of 4 efficient frontiers: unconstrained, long-only,
    +weight cap, +sector cap.

    Shows how constraints shrink the feasible set and shift the frontier.
    """

    @property
    def name(self) -> str:
        return "fig_75_constrained_frontier"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        ann_ret = 252
        ann_risk = np.sqrt(252)

        configs = {
            "Unconstrained": MeanRiskConfig.for_efficient_frontier(size=25),
            "Long-Only": MeanRiskConfig(
                efficient_frontier_size=25,
                min_weights=0.0,
            ),
            "Long-Only + 10% Cap": MeanRiskConfig(
                efficient_frontier_size=25,
                min_weights=0.0,
                max_weights=0.10,
            ),
            "Long-Only + 20% Cap": MeanRiskConfig(
                efficient_frontier_size=25,
                min_weights=0.0,
                max_weights=0.20,
            ),
        }

        colors = ["#9E9E9E", "#2196F3", "#FF9800", "#E91E63"]
        linestyles = ["-", "--", "-.", ":"]

        fig, ax = plt.subplots(figsize=(12, 7))

        for (label, cfg), color, ls in zip(
            configs.items(), colors, linestyles, strict=True
        ):
            ef = build_mean_risk(cfg)
            try:
                population = ef.fit_predict(ret)
                f_returns = np.array(
                    [p.mean for p in population]
                )
                f_risks = np.array(
                    [np.sqrt(p.variance) for p in population]
                )
                ax.plot(
                    f_risks * ann_risk * 100,
                    f_returns * ann_ret * 100,
                    color=color, linestyle=ls, linewidth=2.5, label=label,
                )
            except Exception as e:
                print(f"  Warning: {label} failed: {e}")

        ax.set_xlabel("Annualized Risk (Std Dev, %)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title(
            "How Constraints Reshape the Efficient Frontier",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        print(f"  Fig 75: 4 constrained frontiers, {_N_ASSETS} assets")

        plt.tight_layout()
        self._save(fig)
