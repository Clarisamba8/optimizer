"""Fig90GridSearch — grid search heatmap over L2 coef x correlation threshold."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8


class Fig90GridSearch(FigureGenerator):
    """2D grid search heatmap: L2 regularization vs correlation threshold.

    Uses a simplified grid with direct backtesting (no full GridSearchCV)
    for speed.
    """

    @property
    def name(self) -> str:
        return "fig_90_grid_search"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        # Split into train/test
        split = int(len(ret) * 0.7)
        ret_train = ret.iloc[:split]
        ret_test = ret.iloc[split:]

        l2_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        # Simulate correlation threshold by using different subsets
        # (simplified: just use different numbers of assets)
        n_asset_values = [4, 5, 6, 7, 8]
        n_asset_labels = [f"{n}" for n in n_asset_values]

        sharpe_grid = np.zeros((len(l2_values), len(n_asset_values)))

        for i, l2 in enumerate(l2_values):
            for j, n_a in enumerate(n_asset_values):
                try:
                    sub_ret_train = ret_train.iloc[:, :n_a]
                    sub_ret_test = ret_test.iloc[:, :n_a]

                    cfg = MeanRiskConfig(
                        l2_coef=l2,
                        min_weights=0.0,
                    )
                    opt = build_mean_risk(cfg)
                    opt.fit(sub_ret_train)

                    port_ret = sub_ret_test.values @ opt.weights_
                    ann_ret = np.mean(port_ret) * 252
                    ann_vol = np.std(port_ret) * np.sqrt(252)
                    sr = ann_ret / ann_vol if ann_vol > 0 else 0
                    sharpe_grid[i, j] = sr
                except Exception:
                    sharpe_grid[i, j] = np.nan

        fig, ax = plt.subplots(figsize=(10, 7))

        im = ax.imshow(
            sharpe_grid, cmap="RdYlGn", aspect="auto",
            interpolation="nearest",
        )

        ax.set_xticks(range(len(n_asset_values)))
        ax.set_xticklabels(n_asset_labels, fontsize=9)
        ax.set_yticks(range(len(l2_values)))
        ax.set_yticklabels([f"{v:.3f}" for v in l2_values], fontsize=9)

        ax.set_xlabel("Number of Assets (proxy for correlation threshold)")
        ax.set_ylabel(r"$L_2$ Regularization Coefficient ($\kappa_2$)")

        # Annotate cells with values
        for i in range(len(l2_values)):
            for j in range(len(n_asset_values)):
                val = sharpe_grid[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Out-of-Sample Sharpe Ratio", fontsize=10)

        # Mark optimal cell
        best_idx = np.unravel_index(np.nanargmax(sharpe_grid), sharpe_grid.shape)
        ax.add_patch(mpatches.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor="#E91E63", linewidth=3,
        ))

        ax.set_title(
            "Grid Search Heatmap: Sharpe Ratio Across Parameter Combinations",
            fontsize=12, fontweight="bold",
        )

        print(
            f"  Fig 90: grid search {len(l2_values)}x{len(n_asset_values)}, "
            f"best SR = {np.nanmax(sharpe_grid):.3f}"
        )

        plt.tight_layout()
        self._save(fig)
