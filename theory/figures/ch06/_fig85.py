"""Fig85RollingVsExpanding — rolling vs expanding Sharpe comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.validation import WalkForwardConfig, build_walk_forward, run_cross_val
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8


class Fig85RollingVsExpanding(FigureGenerator):
    """Dual-panel: OOS portfolio Sharpe per walk-forward step under
    rolling vs expanding windows.
    """

    @property
    def name(self) -> str:
        return "fig_85_rolling_vs_expanding"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        opt = build_mean_risk(MeanRiskConfig.for_min_variance())

        # Rolling
        cv_rolling = build_walk_forward(
            WalkForwardConfig(
                train_size=252, test_size=63,
                expend_train=False,
            )
        )
        # Expanding
        cv_expanding = build_walk_forward(
            WalkForwardConfig(
                train_size=252, test_size=63,
                expend_train=True,
            )
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for ax, cv, title, color in [
            (ax1, cv_rolling, "Rolling Window", "#2196F3"),
            (ax2, cv_expanding, "Expanding Window", "#4CAF50"),
        ]:
            try:
                result = run_cross_val(opt, ret, cv=cv)
                oos_returns = pd.Series(result.returns)

                # Compute Sharpe per test window (63 days)
                window = 63
                n_windows = len(oos_returns) // window
                sharpes = []
                for k in range(n_windows):
                    chunk = oos_returns.iloc[k * window: (k + 1) * window]
                    if chunk.std() > 0:
                        sr = chunk.mean() / chunk.std() * np.sqrt(252)
                    else:
                        sr = 0.0
                    sharpes.append(sr)

                x = np.arange(1, len(sharpes) + 1)
                ax.bar(x, sharpes, color=color, alpha=0.7, edgecolor=color)
                ax.axhline(np.mean(sharpes), color="#E91E63", linestyle="--",
                           linewidth=1.5, label=f"Mean SR = {np.mean(sharpes):.2f}")
                ax.set_xlabel("Walk-Forward Step")
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.legend(fontsize=9)
                ax.grid(True, axis="y", alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f"Failed: {e}", transform=ax.transAxes,
                        ha="center", fontsize=9)

        ax1.set_ylabel("Annualized Sharpe Ratio")

        fig.suptitle(
            "Rolling vs Expanding Windows: Stability-Adaptivity Tradeoff",
            fontsize=13, fontweight="bold",
        )

        print(f"  Fig 85: rolling vs expanding, {_N_ASSETS} assets")

        plt.tight_layout()
        self._save(fig)
