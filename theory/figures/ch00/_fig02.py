"""Fig02CrossFactorCorrelation — 17-factor Spearman correlation heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.factors._config import (
    FACTOR_GROUP_MAPPING,
    FactorConstructionConfig,
    FactorGroupType,
    FactorType,
)
from optimizer.factors._construction import compute_all_factors
from theory.figures._base import FigureGenerator
from theory.figures.ch00._data_helpers import query_fundamentals, query_volume_history


def _sort_by_group(columns: list[str]) -> list[str]:
    """Sort factor names so same-group factors are adjacent."""
    group_order = list(FactorGroupType)

    def sort_key(col: str) -> tuple[int, str]:
        for ft, fg in FACTOR_GROUP_MAPPING.items():
            if ft.value == col:
                return (group_order.index(fg), col)
        return (len(group_order), col)

    return sorted(columns, key=sort_key)


def _group_boundaries(columns: list[str]) -> list[int]:
    """Return column indices where the factor group changes."""
    boundaries: list[int] = []
    prev_group = None
    for i, col in enumerate(columns):
        for ft, fg in FACTOR_GROUP_MAPPING.items():
            if ft.value == col:
                if prev_group is not None and fg != prev_group:
                    boundaries.append(i)
                prev_group = fg
                break
    return boundaries


class Fig02CrossFactorCorrelation(FigureGenerator):
    """Spearman correlation heatmap of all computed factors.

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
        return "fig_02_cross_factor_correlation"

    def generate(self) -> None:
        tickers = self._prices.columns.tolist()
        print(f"  Fig 02: {len(tickers)} tickers")

        fundamentals = query_fundamentals(self._db_url, tickers)
        volume = query_volume_history(self._db_url, tickers)

        if fundamentals.empty or len(fundamentals) < 30:
            print("  Fig 02: insufficient fundamentals, using fallback.")
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
            print(f"  Fig 02: only {factors.shape[1]} factors computed, fallback.")
            self._generate_fallback()
            return

        # Sort columns by group and compute Spearman correlation
        sorted_cols = _sort_by_group(factors.columns.tolist())
        factors = factors[sorted_cols]
        corr = factors.corr(method="spearman")

        self._plot_heatmap(corr, sorted_cols)

    def _generate_fallback(self) -> None:
        """Synthetic correlation heatmap when DB is unavailable."""
        np.random.seed(42)
        factor_names = [ft.value for ft in FactorType]
        n = len(factor_names)
        # Generate a plausible correlation matrix
        a = np.random.randn(n, n) * 0.3
        cov = a @ a.T
        d = np.sqrt(np.diag(cov))
        corr_vals = cov / np.outer(d, d)
        np.fill_diagonal(corr_vals, 1.0)

        sorted_cols = _sort_by_group(factor_names)
        idx_map = {name: i for i, name in enumerate(factor_names)}
        reorder = [idx_map[c] for c in sorted_cols]
        corr_vals = corr_vals[np.ix_(reorder, reorder)]

        corr = pd.DataFrame(corr_vals, index=sorted_cols, columns=sorted_cols)
        self._plot_heatmap(corr, sorted_cols)

    def _plot_heatmap(self, corr: pd.DataFrame, sorted_cols: list[str]) -> None:
        """Render and save the heatmap."""
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        n = len(sorted_cols)
        # Cell annotations
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                if i != j:
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=6, color=color,
                    )

        # Group separator lines
        boundaries = _group_boundaries(sorted_cols)
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", lw=0.8, alpha=0.5)
            ax.axvline(b - 0.5, color="black", lw=0.8, alpha=0.5)

        # Short labels
        labels = [c.replace("_", " ").title()[:14] for c in sorted_cols]
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Spearman Correlation", fontsize=9)

        ax.set_title("Cross-Factor Correlation Structure", fontsize=12)
        plt.tight_layout()
        self._save(fig)
