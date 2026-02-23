"""Fig01HysteresisTurnover — Hysteresis reduces universe turnover line chart."""

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

from optimizer.universe._config import HysteresisConfig, InvestabilityScreenConfig
from optimizer.universe._screener import apply_investability_screens
from theory.figures._base import FigureGenerator

_REBALANCE_DAYS = 21  # monthly
_COLORS = ("#E91E63", "#FF9800", "#4CAF50")


def _build_configs() -> list[tuple[str, InvestabilityScreenConfig]]:
    """Build three screening configs: no hysteresis, moderate, wide (default).

    Returns
    -------
    list[tuple[str, InvestabilityScreenConfig]]
        Label and config pairs.
    """
    # No hysteresis: entry == exit for all screens
    no_hyst = InvestabilityScreenConfig(
        market_cap=HysteresisConfig(entry=200_000_000, exit_=200_000_000),
        addv_12m=HysteresisConfig(entry=750_000, exit_=750_000),
        addv_3m=HysteresisConfig(entry=500_000, exit_=500_000),
        trading_frequency=HysteresisConfig(entry=0.95, exit_=0.95),
        price_us=HysteresisConfig(entry=3.0, exit_=3.0),
        price_europe=HysteresisConfig(entry=2.0, exit_=2.0),
        mcap_percentile_entry=0.10,
        mcap_percentile_exit=0.10,
    )
    # Moderate hysteresis: ~10-15% gap
    moderate = InvestabilityScreenConfig(
        market_cap=HysteresisConfig(entry=200_000_000, exit_=175_000_000),
        addv_12m=HysteresisConfig(entry=750_000, exit_=625_000),
        addv_3m=HysteresisConfig(entry=500_000, exit_=425_000),
        trading_frequency=HysteresisConfig(entry=0.95, exit_=0.92),
        price_us=HysteresisConfig(entry=3.0, exit_=2.5),
        price_europe=HysteresisConfig(entry=2.0, exit_=1.75),
        mcap_percentile_entry=0.10,
        mcap_percentile_exit=0.085,
    )
    # Wide hysteresis: default preset (~25% gap)
    wide = InvestabilityScreenConfig()  # uses defaults: 200M/150M etc.

    return [
        ("No Hysteresis", no_hyst),
        ("Moderate (10-15%)", moderate),
        ("Wide (25% default)", wide),
    ]


def _query_fundamentals_and_volume(
    db_url: str,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query fundamentals snapshot and volume history from the database.

    Parameters
    ----------
    db_url:
        SQLAlchemy connection string.
    tickers:
        List of ticker symbols.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (fundamentals DataFrame indexed by ticker with market_cap/current_price,
         volume DataFrame dates x tickers).
    """
    if not _SQLALCHEMY_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()

    try:
        engine = sa.create_engine(db_url)

        # Fundamentals: all active instruments (filter in Python)
        fund_query = sa.text(
            """
            SELECT i.ticker, tp.market_cap, tp.current_price
            FROM instruments i
            JOIN ticker_profiles tp ON tp.instrument_id = i.id
            WHERE i.is_active = true
            """
        )
        with engine.connect() as conn:
            fundamentals = pd.read_sql(fund_query, conn)

        fundamentals = fundamentals[
            fundamentals["ticker"].isin(tickers)
        ]
        fundamentals = fundamentals.drop_duplicates(
            subset="ticker"
        ).set_index("ticker")
        fundamentals["market_cap"] = pd.to_numeric(
            fundamentals["market_cap"], errors="coerce"
        )
        fundamentals["current_price"] = pd.to_numeric(
            fundamentals["current_price"], errors="coerce"
        )

        # Volume history (all active, filter in Python)
        vol_query = sa.text(
            """
            SELECT i.ticker, ph.date, ph.volume
            FROM price_history ph
            JOIN instruments i ON ph.instrument_id = i.id
            WHERE i.is_active = true
              AND ph.volume IS NOT NULL
            """
        )
        with engine.connect() as conn:
            vol_raw = pd.read_sql(
                vol_query, conn, parse_dates=["date"],
            )

        vol_raw = vol_raw[vol_raw["ticker"].isin(tickers)]
        if vol_raw.empty:
            volume = pd.DataFrame()
        else:
            volume = vol_raw.pivot_table(
                index="date",
                columns="ticker",
                values="volume",
            )
            volume.sort_index(inplace=True)

        return fundamentals, volume

    except Exception as exc:
        print(f"  Fig 01: DB query failed ({exc}).")
        return pd.DataFrame(), pd.DataFrame()


class Fig01HysteresisTurnover(FigureGenerator):
    """Line chart comparing monthly universe turnover under three hysteresis regimes.

    Simulates monthly investability screening using real fundamental snapshots
    (market_cap, current_price) and volume history.  Compares: no hysteresis,
    moderate gap, and wide default gap.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        PostgreSQL connection string for querying fundamentals and volume.
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
        return "fig_01_hysteresis_turnover"

    def generate(self) -> None:
        prices = self._prices
        tickers = prices.columns.tolist()
        print(f"  Fig 01: {len(tickers)} tickers in price data")

        fundamentals, volume = _query_fundamentals_and_volume(self._db_url, tickers)

        if fundamentals.empty or volume.empty:
            print("  Fig 01: insufficient DB data, generating fallback.")
            self._generate_fallback()
            return

        # Align volume and prices to common tickers and dates
        common_tickers = (
            prices.columns
            .intersection(volume.columns)
            .intersection(fundamentals.index)
        )
        has_mcap = fundamentals.loc[
            common_tickers, "market_cap"
        ].notna()
        common_tickers = common_tickers[has_mcap]
        if len(common_tickers) < 50:
            msg = f"  Fig 01: only {len(common_tickers)} common"
            print(f"{msg} tickers, using fallback.")
            self._generate_fallback()
            return

        prices_aligned = prices[common_tickers]
        volume_aligned = volume.reindex(columns=common_tickers).fillna(0)
        fund_aligned = fundamentals.loc[common_tickers]
        print(f"  Fig 01: {len(common_tickers)} tickers after alignment")

        # Rebalance dates
        n_dates = len(prices_aligned)
        rebalance_indices = list(range(252, n_dates, _REBALANCE_DAYS))
        if len(rebalance_indices) < 5:
            rebalance_indices = list(range(63, n_dates, _REBALANCE_DAYS))

        configs = _build_configs()
        dates_list: list[pd.Timestamp] = []
        turnover_series: dict[str, list[float]] = {label: [] for label, _ in configs}

        # Per-config independent current_members tracking
        members: dict[str, pd.Index | None] = {label: None for label, _ in configs}

        for idx in rebalance_indices:
            date = prices_aligned.index[idx]
            dates_list.append(date)

            # Use price and volume up to this date
            price_slice = prices_aligned.iloc[:idx + 1]
            vol_slice = volume_aligned.reindex(price_slice.index).fillna(0)

            for label, config in configs:
                prev = members[label]
                universe = apply_investability_screens(
                    fundamentals=fund_aligned,
                    price_history=price_slice,
                    volume_history=vol_slice,
                    config=config,
                    current_members=prev,
                )

                if prev is not None and len(prev) > 0:
                    sym_diff = prev.symmetric_difference(universe)
                    turnover = len(sym_diff) / len(prev) * 100
                else:
                    turnover = 0.0

                turnover_series[label].append(turnover)
                members[label] = universe

        # Remove first date (no previous to compare)
        plot_dates = dates_list[1:]
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for (label, _), color in zip(configs, _COLORS, strict=False):
            values = turnover_series[label][1:]  # skip first (no prev)
            ax.plot(plot_dates, values, color=color, lw=1.5, label=label, alpha=0.85)

        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly Universe Turnover (%)")
        ax.set_title(
            "Hysteresis Reduces Universe Turnover\n"
            f"({len(common_tickers)} instruments, monthly rebalancing)",
        )
        ax.legend(fontsize=9)
        fig.autofmt_xdate()
        plt.tight_layout()
        self._save(fig)

    def _generate_fallback(self) -> None:
        """Generate a synthetic illustration when DB data is unavailable."""
        np.random.seed(42)
        n_months = 60
        dates = pd.date_range("2019-01-01", periods=n_months, freq="ME")

        # Synthetic turnover: no hyst > moderate > wide
        base = np.random.uniform(5, 15, n_months)
        no_hyst = base + np.random.normal(0, 2, n_months)
        moderate = base * 0.65 + np.random.normal(0, 1.5, n_months)
        wide = base * 0.45 + np.random.normal(0, 1.0, n_months)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(dates, np.clip(no_hyst, 0, None), color=_COLORS[0], lw=1.5,
                label="No Hysteresis")
        ax.plot(dates, np.clip(moderate, 0, None), color=_COLORS[1], lw=1.5,
                label="Moderate (10-15%)")
        ax.plot(dates, np.clip(wide, 0, None), color=_COLORS[2], lw=1.5,
                label="Wide (25% default)")

        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly Universe Turnover (%)")
        ax.set_title(
            "Hysteresis Reduces Universe Turnover\n"
            "(Illustrative — real data requires database connection)",
        )
        ax.legend(fontsize=9)
        fig.autofmt_xdate()
        plt.tight_layout()
        self._save(fig)
