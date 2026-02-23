"""Fig74Regularization — L1 vs L2 weight paths as regularization varies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 20


class Fig74Regularization(FigureGenerator):
    """Three-panel bar chart: no reg, L1 reg, L2 reg weight profiles.

    Shows how L1 induces sparsity and L2 compresses weights toward uniform.
    """

    @property
    def name(self) -> str:
        return "fig_74_regularization"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()
        labels = [t.replace("_US_EQ", "").replace("_EQ", "")[:6] for t in top]

        # No regularization
        cfg_none = MeanRiskConfig.for_max_sharpe()
        opt_none = build_mean_risk(cfg_none)
        opt_none.fit(ret)
        w_none = opt_none.weights_

        # L1 regularization (sparse)
        cfg_l1 = MeanRiskConfig(
            objective=cfg_none.objective,
            risk_measure=cfg_none.risk_measure,
            l1_coef=0.05,
            l2_coef=0.0,
        )
        opt_l1 = build_mean_risk(cfg_l1)
        opt_l1.fit(ret)
        w_l1 = opt_l1.weights_

        # L2 regularization (ridge)
        cfg_l2 = MeanRiskConfig(
            objective=cfg_none.objective,
            risk_measure=cfg_none.risk_measure,
            l1_coef=0.0,
            l2_coef=0.05,
        )
        opt_l2 = build_mean_risk(cfg_l2)
        opt_l2.fit(ret)
        w_l2 = opt_l2.weights_

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        x = np.arange(len(labels))

        panels = [
            (axes[0], w_none, "No Regularization", "#9E9E9E"),
            (axes[1], w_l1, r"$L_1$ Regularization ($\kappa_1=0.05$)", "#2196F3"),
            (axes[2], w_l2, r"$L_2$ Regularization ($\kappa_2=0.05$)", "#FF9800"),
        ]

        for ax, w, title, color in panels:
            ax.bar(x, w, color=color, alpha=0.85, edgecolor=color, linewidth=0.5)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.axhline(1.0 / _N_ASSETS, color="#424242", linestyle="--",
                       linewidth=0.8, alpha=0.5, label="Equal weight")
            n_nonzero = np.sum(np.abs(w) > 1e-4)
            ax.text(
                0.95, 0.95,
                f"Non-zero: {n_nonzero}/{_N_ASSETS}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox={"facecolor": "white", "alpha": 0.8},
            )

        axes[0].set_ylabel("Portfolio Weight")

        fig.suptitle(
            "Regularization Effects: L1 Induces Sparsity, L2 Compresses Weights",
            fontsize=12, fontweight="bold",
        )

        print(
            f"  Fig 74: {_N_ASSETS} assets, "
            f"nonzero: none={np.sum(np.abs(w_none) > 1e-4)}, "
            f"L1={np.sum(np.abs(w_l1) > 1e-4)}, "
            f"L2={np.sum(np.abs(w_l2) > 1e-4)}"
        )

        plt.tight_layout()
        self._save(fig)
