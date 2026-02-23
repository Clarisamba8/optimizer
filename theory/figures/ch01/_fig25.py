"""Fig25ParetoFront — dominated vs non-dominated assets scatter."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from skfolio.pre_selection import SelectNonDominated
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns


class Fig25ParetoFront(FigureGenerator):
    """Scatter: dominated (grey) vs non-dominated (blue) using real data.

    Preprocessing pipeline (same as fig24 and fig27):
      prices.ffill() -> prices_to_returns() -> clean_returns() -> dropna()
    No separate completeness pre-filter; clean_returns() already drops any
    column that loses >10% of its observations to DataValidator/OutlierTreater.
    """

    @property
    def name(self) -> str:
        return "fig_25_pareto_front"

    def generate(self) -> None:
        prices = self._prices

        # 3-year window, same as fig24 and fig27
        p3yr = prices.iloc[-756:] if len(prices) > 756 else prices
        returns = clean_returns(prices_to_returns(p3yr.ffill()).dropna()).dropna()

        n_assets = returns.shape[1]
        print(f"  Fig 25: {n_assets} assets x {len(returns)} days (after cleaning)")

        selector = SelectNonDominated()
        selector.fit(returns)
        support = selector.get_support()
        n_nd = support.sum()

        # Annualised moments
        ann_mean = returns.mean() * 252 * 100
        ann_vol = returns.std() * np.sqrt(252) * 100

        fig, ax = plt.subplots(figsize=(9, 6))

        dom_mask = ~support
        ax.scatter(
            ann_vol[dom_mask], ann_mean[dom_mask],
            c="#BDBDBD", s=20, alpha=0.4,
            label=f"Dominated ({int(dom_mask.sum())} assets)",
            zorder=2,
        )
        ax.scatter(
            ann_vol[support], ann_mean[support],
            c="#2196F3", s=50, alpha=0.9,
            label=f"Non-dominated ({int(n_nd)} assets)",
            zorder=3,
        )

        # Pareto step frontier
        nd_vols = ann_vol[support].values
        nd_means = ann_mean[support].values
        order = np.argsort(nd_vols)
        nd_vols_s = nd_vols[order]
        nd_means_s = nd_means[order]
        ax.step(nd_vols_s, nd_means_s,
                where="post", color="#F44336", lw=2, ls="--", zorder=4,
                label="Pareto frontier")

        ax.set_xlabel("Annualised Volatility (%)")
        ax.set_ylabel("Annualised Expected Return (%)")
        ax.set_title(
            "Pareto Front in Mean-Variance Space: Dominated vs Non-Dominated Assets\n"
            f"({n_assets} stocks, NYSE/NASDAQ/LSE/Paris/Xetra, 3-year daily history)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig)
