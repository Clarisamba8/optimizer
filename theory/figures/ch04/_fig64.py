"""Fig64DistanceHeatmaps — 2x2 distance measure heatmaps."""

from __future__ import annotations

import matplotlib.pyplot as plt

from optimizer.optimization import (
    DistanceConfig,
    DistanceType,
    build_distance_estimator,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 20


class Fig64DistanceHeatmaps(FigureGenerator):
    """2x2 heatmaps: Pearson, Kendall, Spearman, Distance Correlation.

    Highlights where different distance measures agree and disagree
    on asset similarity.
    """

    @property
    def name(self) -> str:
        return "fig_64_distance_heatmaps"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()
        labels = [t.replace("_US_EQ", "").replace("_EQ", "")[:6] for t in top]

        distance_types = [
            (DistanceType.PEARSON, "Pearson Distance"),
            (DistanceType.KENDALL, "Kendall Distance"),
            (DistanceType.SPEARMAN, "Spearman Distance"),
            (DistanceType.DISTANCE_CORRELATION, "Distance Correlation"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for ax, (dtype, title) in zip(axes.flat, distance_types, strict=True):
            dist_est = build_distance_estimator(DistanceConfig(distance_type=dtype))
            dist_est.fit(ret)
            dist_matrix = dist_est.distance_

            im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_title(title, fontsize=10, fontweight="bold")

        print(f"  Fig 64: 2x2 distance heatmaps for {_N_ASSETS} assets")

        fig.suptitle(
            "Distance Measure Comparison: Linear vs Non-Linear Codependence",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
