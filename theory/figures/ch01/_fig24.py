"""Fig24CorrelationDistribution — before/after DropCorrelated histograms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skfolio.pre_selection import DropCorrelated
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns


class Fig24CorrelationDistribution(FigureGenerator):
    """Before/after DropCorrelated correlation histograms using real data.

    Uses the full multi-exchange universe with a 3-year window and rho=0.75
    threshold — consistent with the pre-selection pipeline shown in fig27.

    Preprocessing pipeline (same order as fig25 and fig27):
      prices.ffill() -> prices_to_returns() -> clean_returns() -> dropna()
    """

    @property
    def name(self) -> str:
        return "fig_24_correlation_distribution"

    def generate(self) -> None:
        prices = self._prices

        # 3-year window, same as fig25 and fig27
        p3yr = prices.iloc[-756:] if len(prices) > 756 else prices

        # ffill prices (not returns) so genuine gaps -> zero return, then clean
        returns = clean_returns(prices_to_returns(p3yr.ffill()).dropna()).dropna()

        n_assets = returns.shape[1]
        print(f"  Fig 24: {n_assets} assets x {len(returns)} days (full universe)")

        threshold = 0.75  # consistent with fig27 funnel

        # ---- pairwise correlations before filtering ----
        corr_mat = returns.corr().values
        upper_mask = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
        pairwise_before = corr_mat[upper_mask]
        n_above = int((pairwise_before > threshold).sum())
        n_pairs = len(pairwise_before)

        # ---- apply DropCorrelated ----
        drop = DropCorrelated(threshold=threshold)
        drop.fit(returns)
        selected_cols = returns.columns[drop.get_support()]
        df_after = pd.DataFrame(drop.transform(returns), columns=selected_cols)
        n_remaining = df_after.shape[1]
        n_removed = n_assets - n_remaining

        corr_after = df_after.corr().values
        upper_after = np.triu(np.ones(corr_after.shape, dtype=bool), k=1)
        pairwise_after = corr_after[upper_after]
        n_above_after = int((pairwise_after > threshold).sum())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # ---- before ----
        ax1.hist(pairwise_before, bins=80, color="#2196F3", edgecolor="white",
                 lw=0.3, alpha=0.85)
        ax1.axvline(threshold, color="#F44336", lw=2, ls="--",
                    label=f"Threshold rho={threshold}")
        ymax1 = ax1.get_ylim()[1]
        ax1.fill_betweenx([0, ymax1], threshold, 1.0, alpha=0.12, color="#F44336")
        ax1.set_ylim(0, ymax1)
        ax1.set_xlabel("Pairwise Correlation (rho)")
        ax1.set_ylabel("Number of Asset Pairs")
        ax1.set_title(f"Before DropCorrelated\n({n_assets} assets, {n_pairs:,} pairs)")
        ax1.legend(fontsize=9)
        ax1.set_xlim(-1, 1)
        ax1.annotate(
            f"{n_above:,} pairs\nabove threshold",
            xy=(threshold + 0.02, ymax1 * 0.7),
            fontsize=9,
            color="#F44336",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F44336", lw=0.8),
        )

        # ---- after ----
        ax2.hist(pairwise_after, bins=80, color="#4CAF50", edgecolor="white",
                 lw=0.3, alpha=0.85)
        ax2.axvline(threshold, color="#F44336", lw=2, ls="--",
                    label=f"Threshold rho={threshold}")
        ymax2 = ax2.get_ylim()[1]
        ax2.fill_betweenx([0, ymax2], threshold, 1.0, alpha=0.06, color="#F44336")
        ax2.set_ylim(0, ymax2)
        ax2.set_xlabel("Pairwise Correlation (rho)")
        ax2.set_ylabel("Number of Asset Pairs")
        ax2.set_title(
            f"After DropCorrelated\n({n_remaining} assets, {n_removed} removed)"
        )
        ax2.legend(fontsize=9)
        ax2.set_xlim(-1, 1)
        ax2.annotate(
            f"{n_above_after} pairs\nabove threshold",
            xy=(threshold + 0.02, ymax2 * 0.7),
            fontsize=9,
            color="#388E3C",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#388E3C", lw=0.8),
        )

        fig.suptitle(
            f"Pairwise Correlation Distribution Before and After DropCorrelated\n"
            f"(Multi-exchange universe, 3yr window, threshold rho={threshold}, "
            f"N={n_assets} assets)",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
