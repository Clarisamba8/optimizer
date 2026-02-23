"""Fig65WardDendrogram — 30-asset dendrogram with Ward linkage, colored by sector."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    import sqlalchemy as sa

    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False

import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 30
_SECTOR_COLORS = {
    "Technology": "#2196F3",
    "Healthcare": "#4CAF50",
    "Financials": "#FF9800",
    "Consumer Discretionary": "#E91E63",
    "Industrials": "#9C27B0",
    "Energy": "#FF5722",
    "Consumer Staples": "#00BCD4",
    "Materials": "#795548",
    "Utilities": "#607D8B",
    "Real Estate": "#CDDC39",
    "Communication Services": "#F44336",
    "Unknown": "#9E9E9E",
}


def _load_sectors(db_url: str, tickers: list[str]) -> dict[str, str]:
    """Load sector mappings from the database."""
    if not _SQLALCHEMY_AVAILABLE:
        return dict.fromkeys(tickers, "Unknown")
    try:
        engine = sa.create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(
                sa.text(
                    "SELECT ticker, sector FROM stock_info "
                    "WHERE ticker = ANY(:tickers)"
                ),
                {"tickers": tickers},
            )
            mapping = {row[0]: row[1] for row in result}
        return {t: mapping.get(t, "Unknown") for t in tickers}
    except Exception:
        return dict.fromkeys(tickers, "Unknown")


class Fig65WardDendrogram(FigureGenerator):
    """30-asset dendrogram with Ward linkage, colored by sector.

    Uses scipy hierarchical clustering directly for the dendrogram
    visualization, colored by GICS sector from DB.
    """

    def __init__(
        self, prices: pd.DataFrame, output_dir: Path, db_url: str = ""
    ) -> None:
        super().__init__(prices, output_dir)
        self._db_url = db_url

    @property
    def name(self) -> str:
        return "fig_65_ward_dendrogram"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()
        tickers = list(top)
        labels = [t.replace("_US_EQ", "").replace("_EQ", "") for t in tickers]

        # Pearson distance matrix
        corr = ret.corr().values
        dist = np.sqrt(0.5 * (1 - corr))

        # Condensed distance for linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(dist, checks=False)
        z_linkage = linkage(condensed, method="ward")

        # Load sectors for color coding
        sectors = _load_sectors(self._db_url, tickers)
        unique_sectors = sorted(set(sectors.values()))

        # Assign colors to leaf labels based on sector
        sector_color_map = {}
        for s in unique_sectors:
            sector_color_map[s] = _SECTOR_COLORS.get(s, "#9E9E9E")

        fig, ax = plt.subplots(figsize=(14, 7))

        dendrogram(
            z_linkage,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax,
            color_threshold=0.7 * max(z_linkage[:, 2]),
        )

        # Color x-axis labels by sector
        xlbls = ax.get_xticklabels()
        for lbl in xlbls:
            ticker_name = lbl.get_text()
            # Find original ticker
            idx = labels.index(ticker_name) if ticker_name in labels else -1
            if idx >= 0:
                sector = sectors.get(tickers[idx], "Unknown")
                lbl.set_color(sector_color_map.get(sector, "#9E9E9E"))
                lbl.set_fontweight("bold")

        # Legend for sectors present
        import matplotlib.patches as mpatches

        present_sectors = sorted({sectors[t] for t in tickers})
        patches = [
            mpatches.Patch(color=sector_color_map.get(s, "#9E9E9E"), label=s)
            for s in present_sectors
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=2)

        ax.set_ylabel("Ward Linkage Distance")
        ax.set_title(
            "Hierarchical Clustering Dendrogram: "
            "Ward Linkage Recovers Sector Structure",
            fontsize=12, fontweight="bold",
        )

        print(
            f"  Fig 65: dendrogram for {_N_ASSETS} assets,"
            f" {len(present_sectors)} sectors"
        )

        plt.tight_layout()
        self._save(fig)
