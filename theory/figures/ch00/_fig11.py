"""Fig11SectorBalance — Sector balance constraints grouped bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import sqlalchemy as sa
    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False

from optimizer.factors._config import SelectionConfig, SelectionMethod
from optimizer.factors._selection import select_stocks
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 252
_TARGET_COUNT = 100
_COLORS = ("#2196F3", "#E91E63", "#4CAF50")


def _query_sector_and_mcap(
    db_url: str,
    tickers: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Query sector labels and market cap from the database.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (sector_labels indexed by ticker, market_caps indexed by ticker).
        Returns empty Series if DB is unavailable.
    """
    if not _SQLALCHEMY_AVAILABLE:
        return pd.Series(dtype=str), pd.Series(dtype=float)

    try:
        engine = sa.create_engine(db_url)
        # Query all active instruments (filter to tickers in Python)
        query = sa.text(
            """
            SELECT i.ticker, tp.sector, tp.market_cap
            FROM instruments i
            LEFT JOIN ticker_profiles tp
                ON tp.instrument_id = i.id
            WHERE i.is_active = true
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return pd.Series(dtype=str), pd.Series(dtype=float)

        # Filter to requested tickers
        df = df[df["ticker"].isin(tickers)]
        df = df.drop_duplicates(subset="ticker").set_index("ticker")
        sectors = df["sector"].fillna("Unknown")
        mcaps = pd.to_numeric(
            df["market_cap"], errors="coerce"
        ).dropna()
        return sectors, mcaps
    except Exception as exc:
        print(f"  Fig 11: DB query failed ({exc}), fallback.")
        return pd.Series(dtype=str), pd.Series(dtype=float)


class Fig11SectorBalance(FigureGenerator):
    """Grouped bar chart showing sector distribution under three selection modes.

    Compares: (1) unconstrained factor selection, (2) sector-balanced selection,
    and (3) market-cap-weighted proxy (top-100 by market cap).

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        PostgreSQL connection string for querying sector labels and market caps.
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
        return "fig_11_sector_balance"

    def generate(self) -> None:
        prices = self._prices

        # Use most recent window for scoring
        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()
        tickers = returns.columns.tolist()
        n_assets = len(tickers)
        print(f"  Fig 11: {n_assets} assets")

        # Query sector labels and market caps
        sector_labels, mcaps = _query_sector_and_mcap(self._db_url, tickers)

        if sector_labels.empty or len(sector_labels) < 20:
            print("  Fig 11: insufficient sector data, skipping.")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(
                0.5, 0.5, "Insufficient sector data from database",
                transform=ax.transAxes, ha="center", va="center", fontsize=14,
            )
            self._save(fig)
            return

        # Sharpe-rank scores as composite proxy
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        sigma = sigma.replace(0, np.nan)
        scores = (mu / sigma).dropna()

        # Align to tickers with sector data
        common = scores.index.intersection(sector_labels.index)
        scores = scores.loc[common]
        sector_labels = sector_labels.loc[common]
        parent_universe = common

        # 1. Unconstrained selection
        config_unconstrained = SelectionConfig(
            method=SelectionMethod.FIXED_COUNT,
            target_count=_TARGET_COUNT,
            buffer_fraction=0.0,
            sector_balance=False,
        )
        sel_unconstrained = select_stocks(scores, config=config_unconstrained)

        # 2. Sector-balanced selection
        config_balanced = SelectionConfig(
            method=SelectionMethod.FIXED_COUNT,
            target_count=_TARGET_COUNT,
            buffer_fraction=0.0,
            sector_balance=True,
            sector_tolerance=0.05,
        )
        sel_balanced = select_stocks(
            scores,
            config=config_balanced,
            sector_labels=sector_labels,
            parent_universe=parent_universe,
        )

        # 3. Market-cap-weighted proxy: top 100 by market cap
        mcaps_common = mcaps.reindex(common).dropna()
        sel_mcap = mcaps_common.sort_values(ascending=False).index[:_TARGET_COUNT]

        # Count by sector for each selection
        sector_counts_unconstrained = sector_labels.loc[
            sector_labels.index.intersection(sel_unconstrained)
        ].value_counts()
        sector_counts_balanced = sector_labels.loc[
            sector_labels.index.intersection(sel_balanced)
        ].value_counts()
        sector_counts_mcap = sector_labels.loc[
            sector_labels.index.intersection(sel_mcap)
        ].value_counts()

        active_sectors = sorted(set(
            sector_counts_unconstrained.index.tolist()
            + sector_counts_balanced.index.tolist()
            + sector_counts_mcap.index.tolist()
        ))

        counts_u = [sector_counts_unconstrained.get(s, 0) for s in active_sectors]
        counts_b = [sector_counts_balanced.get(s, 0) for s in active_sectors]
        counts_m = [sector_counts_mcap.get(s, 0) for s in active_sectors]

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 5.5))
        x = np.arange(len(active_sectors))
        width = 0.25

        ax.bar(x - width, counts_u, width, label="Unconstrained", color=_COLORS[0])
        ax.bar(x, counts_b, width, label="Sector-Balanced", color=_COLORS[1])
        ax.bar(x + width, counts_m, width, label="Market-Cap Top-100", color=_COLORS[2])

        # Shorten long sector names for display
        short_labels = [s[:15] + "..." if len(s) > 18 else s for s in active_sectors]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Number of Stocks")
        ax.set_title(
            "Sector Balance Constraints Prevent Factor-Driven Concentration\n"
            f"(Top-{_TARGET_COUNT} selection from {n_assets} assets)",
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig)
