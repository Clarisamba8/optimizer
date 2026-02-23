"""Fig37GerberVsPearson — Pearson correlation vs Gerber statistic scatter."""

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

from skfolio.moments import EmpiricalCovariance, GerberCovariance

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 756
_GERBER_THRESHOLD = 0.5

# Colors for same-sector vs cross-sector pairs
_COLOR_SAME  = "#E91E63"  # pink/red — same sector
_COLOR_CROSS = "#2196F3"  # blue — cross-sector


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov:
        Square, symmetric covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix with ones on the diagonal.
    """
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-12, 1.0, std)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr


def _query_sector_mapping(
    db_url: str,
    tickers: list[str],
) -> dict[str, str]:
    """Query the sector for each ticker from the instruments table.

    Returns an empty dict if the DB is unavailable or the query fails.

    Parameters
    ----------
    db_url:
        SQLAlchemy connection string.
    tickers:
        List of ticker symbols to look up.

    Returns
    -------
    dict[str, str]
        Mapping of ticker -> sector string.  Missing tickers get ``"Unknown"``.
    """
    if not _SQLALCHEMY_AVAILABLE:
        return {}

    try:
        engine = sa.create_engine(db_url)
        # Use parameterised query to avoid SQL injection (S608)
        ticker_params = {f"t{i}": v for i, v in enumerate(tickers)}
        placeholders = ", ".join(f":{k}" for k in ticker_params)
        query = sa.text(
            f"""
            SELECT i.ticker, s.name AS sector
            FROM instruments i
            LEFT JOIN sectors s ON i.sector_id = s.id
            WHERE i.ticker IN ({placeholders})
              AND i.is_active = true
            """  # noqa: S608
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=ticker_params)
        return df.set_index("ticker")["sector"].fillna("Unknown").to_dict()
    except Exception as exc:
        print(
            f"  Fig 37: DB query failed ({exc}), "
            "colouring all pairs as cross-sector."
        )
        return {}


class Fig37GerberVsPearson(FigureGenerator):
    """Scatter of Pearson correlation (x) vs Gerber statistic (y) for all pairs.

    Gerber (2022) is a co-movement measure robust to noise and outliers: it
    counts only observations where *both* assets move by at least ``threshold``
    standard deviations in the same or opposite direction, ignoring small
    co-movements that may be noise-driven.

    Points are coloured pink (same sector) vs blue (cross-sector) to show
    that same-sector pairs consistently produce higher Gerber statistics for
    a given Pearson correlation level.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the PNG is saved.
    db_url:
        PostgreSQL connection string used to query sector assignments.
        If the DB is unavailable the figure is still produced but all pairs
        are coloured as cross-sector.
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
        return "fig_37_gerber_vs_pearson"

    def generate(self) -> None:
        prices = self._prices

        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        tickers = returns.columns.tolist()
        n_assets = len(tickers)
        print(f"  Fig 37: {n_assets} assets x {len(returns)} days")

        # Sector mapping (falls back to empty dict if DB is unavailable)
        sector_map = _query_sector_mapping(self._db_url, tickers)

        # Pearson correlation from EmpiricalCovariance
        emp = EmpiricalCovariance()
        emp.fit(returns)
        corr_pearson = _cov_to_corr(emp.covariance_)

        # Gerber statistic from GerberCovariance (already a correlation-like matrix)
        gerber = GerberCovariance(threshold=_GERBER_THRESHOLD)
        gerber.fit(returns)
        corr_gerber = _cov_to_corr(gerber.covariance_)

        # Extract upper-triangle pairs (excluding diagonal)
        idx_i, idx_j = np.triu_indices(n_assets, k=1)
        pearson_vals = corr_pearson[idx_i, idx_j]
        gerber_vals  = corr_gerber[idx_i, idx_j]

        # Determine same/cross-sector for each pair
        is_same_sector = np.array([
            bool(
                sector_map.get(tickers[i], "Unknown_i")
                == sector_map.get(tickers[j], "Unknown_j")
                and sector_map.get(tickers[i], "Unknown_i") != "Unknown"
            )
            for i, j in zip(idx_i, idx_j, strict=False)
        ])

        n_same  = int(is_same_sector.sum())
        n_cross = len(is_same_sector) - n_same
        total_pairs = len(pearson_vals)
        print(
            f"  Pairs: {total_pairs} total "
            f"| {n_same} same-sector | {n_cross} cross-sector"
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        # Cross-sector pairs first (background layer)
        ax.scatter(
            pearson_vals[~is_same_sector],
            gerber_vals[~is_same_sector],
            c=_COLOR_CROSS, s=6, alpha=0.25, linewidths=0,
            label=f"Cross-sector ({n_cross:,} pairs)",
            zorder=2,
        )
        # Same-sector pairs on top
        ax.scatter(
            pearson_vals[is_same_sector],
            gerber_vals[is_same_sector],
            c=_COLOR_SAME, s=12, alpha=0.55, linewidths=0,
            label=f"Same-sector ({n_same:,} pairs)",
            zorder=3,
        )

        # 45-degree reference line (Gerber = Pearson)
        lims = [-1.0, 1.0]
        ax.plot(lims, lims, color="#9E9E9E", lw=1.5, ls="--",
                label="Gerber = Pearson (identity)", zorder=1)
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.axvline(0, color="black", lw=0.6, ls=":")

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Pearson Correlation", fontsize=10)
        ax.set_ylabel(f"Gerber Statistic (threshold={_GERBER_THRESHOLD})", fontsize=10)
        ax.set_title(
            "Gerber Statistic vs Pearson Correlation for All Asset Pairs\n"
            f"(3-year window, {n_assets} assets, {total_pairs:,} pairs — "
            "same-sector pairs highlighted)",
            fontsize=11,
        )
        ax.legend(fontsize=9, markerscale=2.5)
        plt.tight_layout()
        self._save(fig)
