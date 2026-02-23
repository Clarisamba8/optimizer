"""Fig66HRPWeights — dual panel: dendrogram + HRP weight bar chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from optimizer.optimization import HRPConfig, build_hrp
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 12


class Fig66HRPWeights(FigureGenerator):
    """Dual panel: dendrogram (top) + HRP weight bars (bottom).

    Shows how the hierarchical clustering structure maps to the
    final HRP portfolio weights.
    """

    @property
    def name(self) -> str:
        return "fig_66_hrp_weights"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()
        labels = [t.replace("_US_EQ", "").replace("_EQ", "") for t in top]

        # HRP
        hrp = build_hrp(HRPConfig.for_variance())
        hrp.fit(ret)
        w_hrp = hrp.weights_

        # Dendrogram from Pearson distance
        corr = ret.corr().values
        dist = np.sqrt(0.5 * (1 - corr))
        condensed = squareform(dist, checks=False)
        z_linkage = linkage(condensed, method="ward")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 2]},
        )

        # Panel A: Dendrogram
        dend = dendrogram(
            z_linkage, labels=labels, leaf_rotation=45, leaf_font_size=9, ax=ax1,
            color_threshold=0.7 * max(z_linkage[:, 2]),
        )
        ax1.set_ylabel("Ward Distance")
        ax1.set_title("Panel A: Hierarchical Clustering Dendrogram", fontsize=10)

        # Panel B: Weight bar chart (reorder to match dendrogram leaf order)
        leaf_order = dend["leaves"]
        ordered_labels = [labels[i] for i in leaf_order]
        ordered_weights = w_hrp[leaf_order]

        colors = plt.cm.Set2(np.linspace(0, 1, _N_ASSETS))
        ax2.bar(
            range(_N_ASSETS), ordered_weights,
            color=[colors[i] for i in range(_N_ASSETS)],
            edgecolor="#424242", linewidth=0.5,
        )
        ax2.set_xticks(range(_N_ASSETS))
        ax2.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=9)
        ax2.set_ylabel("HRP Weight")
        ax2.set_title("Panel B: HRP Portfolio Weights (dendrogram order)", fontsize=10)

        # Annotate weights
        for i, w in enumerate(ordered_weights):
            ax2.text(i, w + 0.005, f"{w:.1%}", ha="center", fontsize=7)

        print(
            f"  Fig 66: HRP {_N_ASSETS} assets, "
            f"max weight = {w_hrp.max():.2%}, min weight = {w_hrp.min():.2%}"
        )

        fig.suptitle(
            "HRP Recursive Bisection: From Dendrogram to Portfolio Weights",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
