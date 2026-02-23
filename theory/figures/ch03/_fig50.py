"""Fig50FactorBL — two-panel bar chart of factor-level and asset-level BL returns."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_RNG_SEED = 42


class Fig50FactorBL(FigureGenerator):
    """Two-panel bar chart showing factor views propagating to asset returns.

    Panel A: Factor-level posterior expected returns (3 factors).
    Panel B: Asset-level expected returns derived via B * mu_f, showing how
    factor views propagate to all assets through their loadings.

    Uses synthetic factor returns and loadings for pedagogical clarity.
    """

    @property
    def name(self) -> str:
        return "fig_50_factor_bl"

    def generate(self) -> None:
        # Synthetic factor setup
        factor_names = ["Momentum", "Value", "Quality"]
        asset_names = [f"Stock {i+1}" for i in range(8)]
        n_factors = len(factor_names)
        n_assets = len(asset_names)

        # Factor-level posterior returns (annualised %)
        mu_f = np.array([12.0, 5.0, 7.0])

        # Loading matrix (n_assets x n_factors) — diverse exposures
        loadings = np.array([
            [0.8, 0.1, 0.3],   # Stock 1: high momentum
            [0.2, 0.7, 0.4],   # Stock 2: high value
            [0.3, 0.2, 0.9],   # Stock 3: high quality
            [0.6, 0.5, 0.2],   # Stock 4: momentum + value
            [0.1, 0.8, 0.6],   # Stock 5: value + quality
            [0.7, 0.3, 0.7],   # Stock 6: momentum + quality
            [0.4, 0.4, 0.4],   # Stock 7: balanced
            [0.9, 0.1, 0.1],   # Stock 8: pure momentum
        ])

        # Asset-level returns: mu_asset = loadings @ mu_f
        mu_asset = loadings @ mu_f

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                        gridspec_kw={"width_ratios": [1, 2]})

        # Panel A: Factor returns
        colors_f = ["#2196F3", "#4CAF50", "#FF9800"]
        bars_f = ax1.bar(
            range(n_factors), mu_f, color=colors_f, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars_f, mu_f, strict=True):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold",
            )
        ax1.set_xticks(range(n_factors))
        ax1.set_xticklabels(factor_names, fontsize=9)
        ax1.set_ylabel("Annualised Return (%)")
        ax1.set_title(
            "Panel A: Factor-Level\nPosterior Returns",
            fontsize=11, fontweight="bold",
        )
        ax1.set_ylim(0, max(mu_f) * 1.25)

        # Panel B: Asset returns coloured by dominant factor
        dominant_factor = np.argmax(loadings, axis=1)
        asset_colors = [colors_f[d] for d in dominant_factor]

        bars_a = ax2.bar(
            range(n_assets), mu_asset, color=asset_colors, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars_a, mu_asset, strict=True):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, val + 0.15,
                f"{val:.1f}%", ha="center", fontsize=8,
            )
        ax2.set_xticks(range(n_assets))
        ax2.set_xticklabels(asset_names, fontsize=8, rotation=30, ha="right")
        ax2.set_ylabel("Annualised Return (%)")
        ax2.set_title(
            r"Panel B: Asset-Level Returns via $\bar{\mu} = B \bar{\mu}_f$"
            "\nColour = dominant factor exposure",
            fontsize=11, fontweight="bold",
        )
        ax2.set_ylim(0, max(mu_asset) * 1.2)

        # Factor legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=f)
            for c, f in zip(colors_f, factor_names, strict=True)
        ]
        ax2.legend(
            handles=legend_elements, title="Dominant Factor",
            fontsize=8, title_fontsize=9, loc="upper right",
        )

        fig.suptitle(
            "Factor Views Propagate to Asset Returns Through the Loading Matrix",
            fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        self._save(fig)
