"""Fig04RankNormalHistograms — Raw vs z-score vs rank-normal distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from optimizer.factors._config import (
    FactorConstructionConfig,
    FactorType,
    StandardizationConfig,
    StandardizationMethod,
)
from optimizer.factors._construction import compute_all_factors
from optimizer.factors._standardization import standardize_all_factors
from theory.figures._base import FigureGenerator
from theory.figures.ch00._data_helpers import query_fundamentals, query_volume_history

_TARGET_FACTOR = FactorType.BOOK_TO_PRICE.value


class Fig04RankNormalHistograms(FigureGenerator):
    """Three-panel histogram: raw, z-score, and rank-normal distributions.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        PostgreSQL connection string for querying fundamentals.
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
        return "fig_04_rank_normal_histograms"

    def generate(self) -> None:
        tickers = self._prices.columns.tolist()
        print(f"  Fig 04: {len(tickers)} tickers")

        fundamentals = query_fundamentals(self._db_url, tickers)
        volume = query_volume_history(self._db_url, tickers)

        if fundamentals.empty or len(fundamentals) < 30:
            print("  Fig 04: insufficient data, using fallback.")
            self._generate_fallback()
            return

        config = FactorConstructionConfig(factors=(FactorType.BOOK_TO_PRICE,))
        factors = compute_all_factors(
            fundamentals=fundamentals,
            price_history=self._prices,
            volume_history=volume if not volume.empty else None,
            config=config,
        )

        col_missing = _TARGET_FACTOR not in factors.columns
        if col_missing or factors[_TARGET_FACTOR].dropna().shape[0] < 20:
            print("  Fig 04: target factor unavailable, using fallback.")
            self._generate_fallback()
            return

        raw = factors[_TARGET_FACTOR].dropna()

        # Z-score standardization
        zscore_config = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=False,
        )
        zscored, _ = standardize_all_factors(
            factors[[_TARGET_FACTOR]].dropna(), config=zscore_config,
        )
        z_vals = zscored[_TARGET_FACTOR].dropna()

        # Rank-normal standardization
        rn_config = StandardizationConfig(
            method=StandardizationMethod.RANK_NORMAL,
            neutralize_sector=False,
        )
        rn_scored, _ = standardize_all_factors(
            factors[[_TARGET_FACTOR]].dropna(), config=rn_config,
        )
        rn_vals = rn_scored[_TARGET_FACTOR].dropna()

        self._plot(raw, z_vals, rn_vals)

    def _generate_fallback(self) -> None:
        """Synthetic heavy-tailed distribution for illustration."""
        np.random.seed(42)
        raw = pd.Series(np.random.lognormal(0, 0.8, 500), name=_TARGET_FACTOR)
        z_vals = pd.Series((raw - raw.mean()) / raw.std())
        ranks = raw.rank()
        rn_vals = pd.Series(
            sp_stats.norm.ppf((ranks - 0.5) / len(ranks)),
        )
        self._plot(raw, z_vals, rn_vals)

    def _plot(
        self,
        raw: pd.Series,
        z_vals: pd.Series,
        rn_vals: pd.Series,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Panel 1: raw
        axes[0].hist(raw, bins=40, color="#2196F3", alpha=0.8, edgecolor="white")
        axes[0].set_title("Raw Book-to-Price")
        axes[0].set_xlabel("Raw Value")
        axes[0].set_ylabel("Frequency")

        # Panel 2: z-score
        axes[1].hist(z_vals, bins=40, color="#FF9800", alpha=0.8, edgecolor="white")
        axes[1].set_title("Z-Score Standardized")
        axes[1].set_xlabel("Z-Score")

        # Panel 3: rank-normal with N(0,1) overlay
        axes[2].hist(
            rn_vals, bins=40, density=True,
            color="#4CAF50", alpha=0.8, edgecolor="white",
        )
        x = np.linspace(-3.5, 3.5, 200)
        axes[2].plot(x, sp_stats.norm.pdf(x), "k--", lw=1.5, label="N(0,1)")
        axes[2].set_title("Rank-Normal Transformed")
        axes[2].set_xlabel("Rank-Normal Score")
        axes[2].legend(fontsize=8)

        fig.suptitle(
            "Rank-Normal Transformation Tames Heavy Tails\n"
            f"({len(raw)} stocks, book-to-price ratio)",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        self._save(fig)
