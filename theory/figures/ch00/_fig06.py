"""Fig06AlphaScoreDistribution — Composite score histogram with quintile coloring."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from optimizer.factors._config import (
    CompositeScoringConfig,
    FactorConstructionConfig,
    StandardizationConfig,
)
from optimizer.factors._construction import compute_all_factors
from optimizer.factors._scoring import compute_composite_score
from optimizer.factors._standardization import standardize_all_factors
from theory.figures._base import FigureGenerator
from theory.figures.ch00._data_helpers import query_fundamentals, query_volume_history


class Fig06AlphaScoreDistribution(FigureGenerator):
    """Histogram of composite AlphaScore colored by quintile.

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
        return "fig_06_alpha_score_distribution"

    def generate(self) -> None:
        tickers = self._prices.columns.tolist()
        print(f"  Fig 06: {len(tickers)} tickers")

        fundamentals = query_fundamentals(self._db_url, tickers)
        volume = query_volume_history(self._db_url, tickers)

        if fundamentals.empty or len(fundamentals) < 30:
            print("  Fig 06: insufficient data, using fallback.")
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
            print("  Fig 06: too few factors, using fallback.")
            self._generate_fallback()
            return

        std_config = StandardizationConfig(neutralize_sector=False)
        standardized, coverage = standardize_all_factors(factors, config=std_config)

        composite = compute_composite_score(
            standardized, coverage, config=CompositeScoringConfig(),
        )
        composite = composite.dropna()

        if len(composite) < 20:
            print("  Fig 06: too few scores, using fallback.")
            self._generate_fallback()
            return

        self._plot(composite)

    def _generate_fallback(self) -> None:
        """Synthetic AlphaScore distribution."""
        np.random.seed(42)
        # Slight positive skew from momentum component
        scores = np.random.normal(0, 1, 500) + 0.1 * np.random.exponential(1, 500)
        composite = pd.Series(scores)
        self._plot(composite)

    def _plot(self, composite: pd.Series) -> None:
        q20 = composite.quantile(0.20)
        q80 = composite.quantile(0.80)

        fig, ax = plt.subplots(figsize=(10, 5.5))

        # Split into quintile regions for coloring
        bottom = composite[composite <= q20]
        middle = composite[(composite > q20) & (composite <= q80)]
        top = composite[composite > q80]

        bins = np.linspace(composite.min(), composite.max(), 50)
        ax.hist(bottom, bins=bins, color="#E91E63", alpha=0.8, label="Bottom 20%")
        ax.hist(middle, bins=bins, color="#9E9E9E", alpha=0.6, label="Middle 60%")
        ax.hist(top, bins=bins, color="#4CAF50", alpha=0.8, label="Top 20%")

        # Quantile boundaries
        for q_val, label in [(q20, "Q20"), (q80, "Q80")]:
            ax.axvline(q_val, color="black", ls="--", lw=1.2, alpha=0.7)
            ax.text(
                q_val, ax.get_ylim()[1] * 0.92, f" {label}={q_val:.2f}",
                fontsize=8, va="top",
            )

        # Normal density overlay
        x = np.linspace(composite.min(), composite.max(), 200)
        pdf = sp_stats.norm.pdf(x, composite.mean(), composite.std())
        ax2 = ax.twinx()
        ax2.plot(x, pdf, "k--", lw=1.2, alpha=0.5, label="Normal fit")
        ax2.set_ylabel("Density", fontsize=9)
        ax2.set_ylim(0, pdf.max() * 2.5)

        ax.set_xlabel("Composite AlphaScore")
        ax.set_ylabel("Frequency")
        ax.set_title(
            "Cross-Sectional Distribution of Composite Alpha Score\n"
            f"({len(composite)} investable stocks)",
        )
        ax.legend(fontsize=9, loc="upper left")
        plt.tight_layout()
        self._save(fig)
