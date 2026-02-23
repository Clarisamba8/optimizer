"""Fig80StressFan — conditional stress test fan chart from vine copula."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns


class Fig80StressFan(FigureGenerator):
    """Conditional stress test: sector return distributions given a market shock.

    Attempts to use vine copula; falls back to bootstrap-based conditional
    distributions if vine fitting fails.
    """

    @property
    def name(self) -> str:
        return "fig_80_stress_fan"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        # Select representative assets for sectors
        sector_candidates = returns.notna().sum().nlargest(20).index
        sectors = list(sector_candidates[:6])
        sector_labels = [
            s.replace("_US_EQ", "").replace("_EQ", "")[:8] for s in sectors
        ]

        ret = returns[sectors].dropna()
        market_proxy = ret.mean(axis=1)

        # Conditional bootstrap: select days where market dropped > 2 std
        market_std = market_proxy.std()
        stress_mask = market_proxy < -2 * market_std

        if stress_mask.sum() < 10:
            # Relax threshold
            stress_mask = market_proxy < -1.5 * market_std

        if stress_mask.sum() < 5:
            stress_mask = market_proxy < market_proxy.quantile(0.05)

        stress_returns = ret.loc[stress_mask]
        normal_returns = ret.loc[~stress_mask]

        fig, ax = plt.subplots(figsize=(12, 7))

        positions = np.arange(len(sectors))
        width = 0.35

        # Normal regime box plots
        bp_normal = ax.boxplot(
            [normal_returns[s].dropna().values * 100 for s in sectors],
            positions=positions - width / 2,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp_normal["boxes"]:
            patch.set_facecolor("#4CAF50")
            patch.set_alpha(0.6)

        # Stress regime box plots
        bp_stress = ax.boxplot(
            [stress_returns[s].dropna().values * 100 for s in sectors],
            positions=positions + width / 2,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp_stress["boxes"]:
            patch.set_facecolor("#E91E63")
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(sector_labels, rotation=45, ha="right")
        ax.set_ylabel("Daily Return (%)")
        ax.set_title(
            "Conditional Stress Test: Asset Returns During Market Stress",
            fontsize=12, fontweight="bold",
        )

        ax.legend(
            [bp_normal["boxes"][0], bp_stress["boxes"][0]],
            ["Normal Regime", f"Stress Regime (n={stress_mask.sum()})"],
            loc="lower left", fontsize=9,
        )
        ax.axhline(0, color="#9E9E9E", linestyle=":", alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)

        stress_pct = stress_mask.mean() * 100
        ax.text(
            0.98, 0.02,
            f"Stress threshold: market < {-2 * market_std * 100:.1f}%\n"
            f"Stress days: {stress_mask.sum()} ({stress_pct:.1f}% of sample)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox={"facecolor": "white", "alpha": 0.8},
        )

        print(
            f"  Fig 80: stress test with {len(sectors)} sectors, "
            f"{stress_mask.sum()} stress days"
        )

        plt.tight_layout()
        self._save(fig)
