"""Fig08VIFBarChart — Variance inflation factor horizontal bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.factors._config import (
    FactorConstructionConfig,
    StandardizationConfig,
)
from optimizer.factors._construction import compute_all_factors
from optimizer.factors._standardization import standardize_all_factors
from optimizer.factors._validation import compute_vif
from theory.figures._base import FigureGenerator
from theory.figures.ch00._data_helpers import query_fundamentals, query_volume_history


class Fig08VIFBarChart(FigureGenerator):
    """Horizontal bar chart of VIF values with threshold references.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        PostgreSQL connection string.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        output_dir: Path,
        db_url: str,
    ) -> None:
        super().__init__(prices, output_dir)
        self._db_url = db_url

    @property
    def name(self) -> str:
        return "fig_08_vif_bar_chart"

    def generate(self) -> None:
        tickers = self._prices.columns.tolist()
        print(f"  Fig 08: {len(tickers)} tickers")

        fundamentals = query_fundamentals(self._db_url, tickers)
        volume = query_volume_history(self._db_url, tickers)

        if fundamentals.empty or len(fundamentals) < 30:
            print("  Fig 08: insufficient data, using fallback.")
            self._generate_fallback()
            return

        config = FactorConstructionConfig.for_all_factors()
        factors = compute_all_factors(
            fundamentals=fundamentals,
            price_history=self._prices,
            volume_history=volume if not volume.empty else None,
            config=config,
        )

        if factors.shape[1] < 3:
            print("  Fig 08: too few factors, using fallback.")
            self._generate_fallback()
            return

        std_config = StandardizationConfig(neutralize_sector=False)
        standardized, _ = standardize_all_factors(factors, config=std_config)
        clean = standardized.dropna()

        if len(clean) < 10:
            print("  Fig 08: too few clean rows, using fallback.")
            self._generate_fallback()
            return

        vif = compute_vif(clean)
        self._plot(vif)

    def _generate_fallback(self) -> None:
        """Synthetic VIF values for illustration."""
        from optimizer.factors._config import FactorType

        np.random.seed(42)
        factor_names = [ft.value for ft in FactorType]
        # Simulate plausible VIF values
        vif_vals = np.random.lognormal(0.5, 0.6, len(factor_names))
        vif_vals = np.clip(vif_vals, 1.0, 25.0)
        vif = pd.Series(vif_vals, index=factor_names)
        self._plot(vif)

    def _plot(self, vif: pd.Series) -> None:
        vif_sorted = vif.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(5, len(vif_sorted) * 0.35)))

        # Color by threshold
        colors = []
        for v in vif_sorted.values:
            if v > 10:
                colors.append("#E91E63")
            elif v > 5:
                colors.append("#FF9800")
            else:
                colors.append("#4CAF50")

        ax.barh(
            range(len(vif_sorted)), vif_sorted.values,
            color=colors, alpha=0.85,
        )

        # Labels
        labels = [n.replace("_", " ").title()[:20] for n in vif_sorted.index]
        ax.set_yticks(range(len(vif_sorted)))
        ax.set_yticklabels(labels, fontsize=8)

        # Value annotations
        for i, v in enumerate(vif_sorted.values):
            ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=8)

        # Threshold lines
        ax.axvline(5, color="#FF9800", ls="--", lw=1.2, alpha=0.8)
        ax.text(5.1, len(vif_sorted) - 0.5, "Concern (5)", fontsize=8,
                color="#FF9800", va="bottom")
        ax.axvline(10, color="#E91E63", ls="--", lw=1.2, alpha=0.8)
        ax.text(10.1, len(vif_sorted) - 0.5, "Critical (10)", fontsize=8,
                color="#E91E63", va="bottom")

        ax.set_xlabel("Variance Inflation Factor")
        ax.set_title("Variance Inflation Factors Identify Redundant Factors")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4CAF50", label="VIF < 5 (OK)"),
            Patch(facecolor="#FF9800", label="5 < VIF < 10 (Concern)"),
            Patch(facecolor="#E91E63", label="VIF > 10 (Critical)"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

        plt.tight_layout()
        self._save(fig)
