"""Fig68HERCvsHRP — side-by-side weight and risk contribution comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.optimization import HERCConfig, HRPConfig, build_herc, build_hrp
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 15


class Fig68HERCvsHRP(FigureGenerator):
    """Side-by-side weight and RC bars: HERC vs HRP.

    Shows that HERC achieves more balanced risk contributions
    within clusters compared to HRP.
    """

    def __init__(
        self, prices: pd.DataFrame, output_dir: Path, db_url: str = ""
    ) -> None:
        super().__init__(prices, output_dir)
        self._db_url = db_url

    @property
    def name(self) -> str:
        return "fig_68_herc_vs_hrp"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()
        labels = [t.replace("_US_EQ", "").replace("_EQ", "") for t in top]

        # Fit HRP and HERC
        hrp = build_hrp(HRPConfig.for_variance())
        hrp.fit(ret)
        w_hrp = hrp.weights_

        herc = build_herc(HERCConfig.for_variance())
        herc.fit(ret)
        w_herc = herc.weights_

        # Compute risk contributions
        cov = ret.cov().values

        def risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            marginal = sigma @ w
            rc = w * marginal
            total = rc.sum()
            return rc / total if total > 0 else rc

        rc_hrp = risk_contributions(w_hrp, cov)
        rc_herc = risk_contributions(w_herc, cov)

        # Measure balance: coefficient of variation of RC
        cv_hrp = np.std(rc_hrp) / np.mean(rc_hrp) if np.mean(rc_hrp) > 0 else 0
        cv_herc = np.std(rc_herc) / np.mean(rc_herc) if np.mean(rc_herc) > 0 else 0

        print(
            f"  Fig 68: {_N_ASSETS} assets, "
            f"HRP RC CV = {cv_hrp:.2f}, HERC RC CV = {cv_herc:.2f}"
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        x = np.arange(len(labels))

        # Row 1: Weights
        axes[0, 0].bar(
            x, w_hrp, color="#2196F3", alpha=0.8, edgecolor="#1565C0", linewidth=0.5,
        )
        axes[0, 0].set_title("HRP Weights", fontsize=10, fontweight="bold")
        axes[0, 0].set_ylabel("Weight")

        axes[0, 1].bar(
            x, w_herc, color="#FF5722", alpha=0.8, edgecolor="#BF360C", linewidth=0.5,
        )
        axes[0, 1].set_title("HERC Weights", fontsize=10, fontweight="bold")

        # Row 2: Risk contributions
        axes[1, 0].bar(
            x, rc_hrp, color="#2196F3", alpha=0.6, edgecolor="#1565C0", linewidth=0.5,
        )
        axes[1, 0].axhline(
            1.0 / _N_ASSETS, color="#E91E63", linestyle="--", linewidth=1.5,
            label=f"Equal = {1/_N_ASSETS:.1%}",
        )
        axes[1, 0].set_title(f"HRP Risk Contributions (CV = {cv_hrp:.2f})", fontsize=10)
        axes[1, 0].set_ylabel("Risk Contribution")
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].bar(
            x, rc_herc, color="#FF5722", alpha=0.6, edgecolor="#BF360C", linewidth=0.5,
        )
        axes[1, 1].axhline(
            1.0 / _N_ASSETS, color="#E91E63", linestyle="--", linewidth=1.5,
            label=f"Equal = {1/_N_ASSETS:.1%}",
        )
        axes[1, 1].set_title(
            f"HERC Risk Contributions (CV = {cv_herc:.2f})", fontsize=10,
        )
        axes[1, 1].legend(fontsize=8)

        for ax in axes.flat:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)

        fig.suptitle(
            "HERC vs HRP: Weight and Risk Contribution Comparison",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
