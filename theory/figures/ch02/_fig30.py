"""Fig30ShrinkageScatter — raw sample means vs James-Stein shrunk means."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import EmpiricalMu, ShrunkMu

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

# Use 3-year rolling window for moment estimation (consistent with ch01)
_WINDOW_DAYS = 756
# Cap the number of assets for a legible scatter: individual arrows require ~20-80 assets.
# With N=2255 all points collapse onto the identity line and arrows overlap completely.
_MAX_SCATTER_ASSETS = 60


class Fig30ShrinkageScatter(FigureGenerator):
    """Scatter of raw EmpiricalMu (x-axis) vs James-Stein ShrunkMu (y-axis).

    Draws the 45-degree identity line (no shrinkage reference) and annotates
    shrinkage arrows from each raw mean to its shrunk counterpart so the
    reader can visually follow the pull toward the grand mean.

    Both estimators are fitted on the same 3-year window of clean arithmetic
    returns.  All values are annualised (× 252 for means) and expressed as
    percentages for readability.
    """

    @property
    def name(self) -> str:
        return "fig_30_shrinkage_scatter"

    def generate(self) -> None:
        prices = self._prices

        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        # Keep only assets with annual vol in [15%, 45%] to avoid degenerate outliers,
        # then cap at _MAX_SCATTER_ASSETS so shrinkage arrows are individually visible.
        annual_vol = returns.std() * np.sqrt(252)
        vol_mask = (annual_vol >= 0.15) & (annual_vol <= 0.45)
        filtered = returns.loc[:, vol_mask]
        if filtered.shape[1] < 20:
            filtered = returns
        returns = filtered.iloc[:, :_MAX_SCATTER_ASSETS]

        n_assets = returns.shape[1]
        print(f"  Fig 30: {n_assets} assets x {len(returns)} days")

        emp = EmpiricalMu()
        emp.fit(returns)
        mu_raw = emp.mu_ * 252 * 100  # annualised %

        shrunk = ShrunkMu()
        shrunk.fit(returns)
        mu_shrunk = shrunk.mu_ * 252 * 100  # annualised %

        # Axis limits with 10% margin
        all_vals = np.concatenate([mu_raw, mu_shrunk])
        lo = all_vals.min() - abs(all_vals.min()) * 0.15
        hi = all_vals.max() + abs(all_vals.max()) * 0.15

        fig, ax = plt.subplots(figsize=(8, 7))

        # 45-degree identity line (no shrinkage)
        ax.plot([lo, hi], [lo, hi], color="#9E9E9E", lw=1.5, ls="--",
                label="y = x  (no shrinkage)", zorder=1)

        # Grand mean reference line (shrinkage target)
        grand_mean = float(mu_raw.mean())
        ax.axhline(grand_mean, color="#FF9800", lw=1.0, ls=":",
                   label=f"Grand mean = {grand_mean:.1f}%", zorder=1)
        ax.axvline(grand_mean, color="#FF9800", lw=1.0, ls=":", zorder=1)

        # Shrinkage arrows
        for x_val, y_val in zip(mu_raw, mu_shrunk, strict=False):
            ax.annotate(
                "",
                xy=(y_val, y_val),
                xytext=(x_val, y_val),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#E91E63",
                    "lw": 0.8,
                    "alpha": 0.55,
                },
                zorder=2,
            )

        # Scatter points
        ax.scatter(
            mu_raw, mu_shrunk,
            c="#2196F3", s=45, alpha=0.85, zorder=3,
            label=f"Assets (n={n_assets})",
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Raw Sample Mean — EmpiricalMu (annualised %)", fontsize=10)
        ax.set_ylabel("Shrunk Mean — James-Stein ShrunkMu (annualised %)", fontsize=10)
        ax.set_title(
            "James-Stein Shrinkage: Raw Sample Means vs Shrunk Means\n"
            f"(3-year window, {n_assets} assets — arrows show pull toward grand mean)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig)
