"""Fig05SectorNeutralization — Before/after sector neutralization box plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.factors._config import (
    FactorConstructionConfig,
    FactorType,
    StandardizationConfig,
    StandardizationMethod,
)
from optimizer.factors._construction import compute_all_factors
from optimizer.factors._standardization import standardize_all_factors
from theory.figures._base import FigureGenerator
from theory.figures.ch00._data_helpers import (
    query_fundamentals,
    query_sector_labels,
    query_volume_history,
)

_TARGET_FACTOR = FactorType.BOOK_TO_PRICE.value


class Fig05SectorNeutralization(FigureGenerator):
    """Two-panel box plots showing sector neutralization effect.

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
        return "fig_05_sector_neutralization"

    def generate(self) -> None:
        tickers = self._prices.columns.tolist()
        print(f"  Fig 05: {len(tickers)} tickers")

        fundamentals = query_fundamentals(self._db_url, tickers)
        sector_labels = query_sector_labels(self._db_url, tickers)
        volume = query_volume_history(self._db_url, tickers)

        if fundamentals.empty or sector_labels.empty or len(fundamentals) < 30:
            print("  Fig 05: insufficient data, using fallback.")
            self._generate_fallback()
            return

        config = FactorConstructionConfig(factors=(FactorType.BOOK_TO_PRICE,))
        factors = compute_all_factors(
            fundamentals=fundamentals,
            price_history=self._prices,
            volume_history=volume if not volume.empty else None,
            config=config,
        )

        if _TARGET_FACTOR not in factors.columns:
            print("  Fig 05: target factor unavailable, using fallback.")
            self._generate_fallback()
            return

        raw_df = factors[[_TARGET_FACTOR]].dropna()
        common = raw_df.index.intersection(sector_labels.index)
        if len(common) < 30:
            print("  Fig 05: insufficient overlap, using fallback.")
            self._generate_fallback()
            return

        raw_df = raw_df.loc[common]
        sectors = sector_labels.loc[common]

        # Before neutralization (z-score, no sector neutralization)
        before_config = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=False,
        )
        before, _ = standardize_all_factors(raw_df, config=before_config)

        # After neutralization
        after_config = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=True,
        )
        after, _ = standardize_all_factors(
            raw_df, config=after_config, sector_labels=sectors,
        )

        self._plot(before[_TARGET_FACTOR], after[_TARGET_FACTOR], sectors)

    def _generate_fallback(self) -> None:
        """Synthetic sector-biased data for illustration."""
        np.random.seed(42)
        sectors_list = ["Financials", "Technology", "Healthcare", "Energy", "Utilities",
                        "Consumer", "Industrials"]
        n_per = 50
        tickers = [f"T{i}" for i in range(n_per * len(sectors_list))]
        sector_data = []
        before_data = []
        after_data = []

        for i, sector in enumerate(sectors_list):
            bias = (i - 3) * 0.6  # financials high, tech low
            vals = np.random.normal(bias, 1.0, n_per)
            sector_data.extend([sector] * n_per)
            before_data.extend(vals.tolist())
            neutralized = vals - np.mean(vals)
            after_data.extend(neutralized.tolist())

        sectors = pd.Series(sector_data, index=tickers)
        before = pd.Series(before_data, index=tickers)
        after = pd.Series(after_data, index=tickers)
        self._plot(before, after, sectors)

    def _plot(
        self,
        before: pd.Series,
        after: pd.Series,
        sectors: pd.Series,
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

        # Get unique sectors sorted by median before-score (descending)
        sector_medians = before.groupby(sectors).median().sort_values(ascending=False)
        sector_order = sector_medians.index.tolist()

        # Before
        bp_data_before = [before[sectors == s].dropna().values for s in sector_order]
        bp1 = ax1.boxplot(
            bp_data_before, labels=sector_order, patch_artist=True, vert=True,
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor("#2196F3")
            patch.set_alpha(0.7)
        ax1.set_title("Before Sector Neutralization")
        ax1.set_ylabel("Z-Score")
        ax1.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax1.tick_params(axis="x", rotation=35)

        # After
        bp_data_after = [after[sectors == s].dropna().values for s in sector_order]
        bp2 = ax2.boxplot(
            bp_data_after, labels=sector_order, patch_artist=True, vert=True,
        )
        for patch in bp2["boxes"]:
            patch.set_facecolor("#4CAF50")
            patch.set_alpha(0.7)
        ax2.set_title("After Sector Neutralization")
        ax2.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax2.tick_params(axis="x", rotation=35)

        fig.suptitle(
            "Sector Neutralization Removes Structural Biases\n"
            f"(Book-to-Price, {len(before)} stocks)",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        self._save(fig)
