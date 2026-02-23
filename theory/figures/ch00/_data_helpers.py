"""Shared database query helpers for Chapter 00 figure generators."""

from __future__ import annotations

import pandas as pd

try:
    import sqlalchemy as sa

    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False


def query_fundamentals(
    db_url: str,
    tickers: list[str],
) -> pd.DataFrame:
    """Query fundamental data needed for ``compute_all_factors()``.

    Returns a DataFrame indexed by ticker with columns mapped to the
    ``fundamentals`` argument of :func:`optimizer.factors.compute_all_factors`:
    ``market_cap``, ``book_value``, ``net_income``, ``total_revenue``,
    ``gross_profit``, ``ebitda``, ``enterprise_value``,
    ``operating_cashflow``, ``dividend_yield``,
    ``shares_outstanding``, ``current_price``.
    ``total_equity`` and ``total_assets`` are derived from available
    columns when possible.

    Parameters
    ----------
    db_url:
        SQLAlchemy connection string.
    tickers:
        Ticker symbols to include.

    Returns
    -------
    pd.DataFrame
        Fundamentals indexed by ticker. Empty if DB is unavailable.
    """
    if not _SQLALCHEMY_AVAILABLE:
        return pd.DataFrame()

    try:
        engine = sa.create_engine(db_url)
        query = sa.text(
            """
            SELECT
                i.ticker,
                tp.market_cap,
                tp.current_price,
                tp.book_value,
                tp.trailing_eps,
                tp.shares_outstanding,
                tp.total_revenue,
                tp.gross_profits   AS gross_profit,
                tp.ebitda,
                tp.enterprise_value,
                tp.operating_cashflow,
                tp.dividend_yield,
                tp.total_cash,
                tp.total_debt,
                tp.debt_to_equity,
                tp.profit_margins,
                tp.return_on_equity
            FROM instruments i
            JOIN ticker_profiles tp ON tp.instrument_id = i.id
            WHERE i.is_active = true
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        df = df[df["ticker"].isin(tickers)]
        df = df.drop_duplicates(subset="ticker").set_index("ticker")

        # Convert to numeric
        numeric_cols = [
            "market_cap", "current_price", "book_value", "trailing_eps",
            "shares_outstanding", "total_revenue", "gross_profit", "ebitda",
            "enterprise_value", "operating_cashflow", "dividend_yield",
            "total_cash", "total_debt", "debt_to_equity",
            "profit_margins", "return_on_equity",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # The DB stores book_value and trailing_eps as per-share values,
        # but compute_all_factors expects totals (book_value / market_cap).
        # Scale per-share -> total by multiplying by shares_outstanding.
        shares = df.get("shares_outstanding")
        if shares is not None:
            if "book_value" in df.columns:
                df["book_value"] = df["book_value"] * shares
            if "trailing_eps" in df.columns:
                df["net_income"] = df["trailing_eps"] * shares

        # Derive total_equity (same as total book value)
        if "book_value" in df.columns:
            df["total_equity"] = df["book_value"]

        # Derive total_assets from total_equity + total_debt
        if "total_equity" in df.columns:
            debt = df.get("total_debt")
            if debt is None and "debt_to_equity" in df.columns:
                debt = df["debt_to_equity"] / 100 * df["total_equity"]
            if debt is not None:
                df["total_assets"] = df["total_equity"] + debt

        return df

    except Exception as exc:
        print(f"  _data_helpers: fundamentals query failed ({exc}).")
        return pd.DataFrame()


def query_sector_labels(
    db_url: str,
    tickers: list[str],
) -> pd.Series:
    """Query sector labels from the database.

    Parameters
    ----------
    db_url:
        SQLAlchemy connection string.
    tickers:
        Ticker symbols to include.

    Returns
    -------
    pd.Series
        Sector label per ticker.  Empty if DB is unavailable.
    """
    if not _SQLALCHEMY_AVAILABLE:
        return pd.Series(dtype=str)

    try:
        engine = sa.create_engine(db_url)
        query = sa.text(
            """
            SELECT i.ticker, tp.sector
            FROM instruments i
            LEFT JOIN ticker_profiles tp ON tp.instrument_id = i.id
            WHERE i.is_active = true
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        df = df[df["ticker"].isin(tickers)]
        df = df.drop_duplicates(subset="ticker").set_index("ticker")
        return df["sector"].fillna("Unknown")

    except Exception as exc:
        print(f"  _data_helpers: sector query failed ({exc}).")
        return pd.Series(dtype=str)


def query_volume_history(
    db_url: str,
    tickers: list[str],
) -> pd.DataFrame:
    """Query daily volume history from the database.

    Parameters
    ----------
    db_url:
        SQLAlchemy connection string.
    tickers:
        Ticker symbols to include.

    Returns
    -------
    pd.DataFrame
        Dates x tickers volume matrix.  Empty if DB is unavailable.
    """
    if not _SQLALCHEMY_AVAILABLE:
        return pd.DataFrame()

    try:
        engine = sa.create_engine(db_url)
        query = sa.text(
            """
            SELECT i.ticker, ph.date, ph.volume
            FROM price_history ph
            JOIN instruments i ON ph.instrument_id = i.id
            WHERE i.is_active = true
              AND ph.volume IS NOT NULL
            """
        )
        with engine.connect() as conn:
            vol_raw = pd.read_sql(query, conn, parse_dates=["date"])

        vol_raw = vol_raw[vol_raw["ticker"].isin(tickers)]
        if vol_raw.empty:
            return pd.DataFrame()

        volume = vol_raw.pivot_table(
            index="date", columns="ticker", values="volume",
        )
        volume.sort_index(inplace=True)
        return volume

    except Exception as exc:
        print(f"  _data_helpers: volume query failed ({exc}).")
        return pd.DataFrame()
