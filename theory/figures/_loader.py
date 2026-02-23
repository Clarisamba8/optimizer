"""Price data loader — single responsibility: DB access and price loading."""

from __future__ import annotations

import pandas as pd
import sqlalchemy as sa
import exchange_calendars as xcals

# Same MIC mapping as api/app/services/trading_calendar.py
EXCHANGE_NAME_TO_MIC: dict[str, str] = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS",
    "London Stock Exchange": "XLON",
    "Euronext Paris": "XPAR",
    "Deutsche Börse Xetra": "XFRA",
}


class PriceLoader:
    """Load adjusted close prices from a PostgreSQL database.

    Responsibilities
    ----------------
    - Execute the DB query
    - Build ticker-to-exchange mapping
    - Calendar-aware forward-fill (non-trading-day gaps only)
    - Filter by minimum genuine trading day count

    Parameters
    ----------
    db_url:
        SQLAlchemy-compatible connection string.
    min_days:
        Minimum number of genuine (non-gap) trading days required to include
        a ticker. Default 1200 (~5 years).
    max_assets:
        If set, truncate to the first *max_assets* columns after filtering.
        Kept on the class to allow construction-time configuration while
        still exposing the parameter on ``load()`` for ad-hoc overrides.
    """

    def __init__(
        self,
        db_url: str,
        min_days: int = 1200,
        max_assets: int | None = None,
    ) -> None:
        self._db_url = db_url
        self._min_days = min_days
        self._max_assets = max_assets

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, max_assets: int | None = None) -> pd.DataFrame:
        """Load and return a wide price DataFrame (date x ticker).

        Calendar gaps (NaN on a non-trading day for a given exchange) are
        forward-filled using exchange_calendars to distinguish them from
        genuine missing data.  Only instruments with at least ``min_days``
        trading days are included.

        Parameters
        ----------
        max_assets:
            If set, overrides the instance-level ``max_assets`` and truncates
            to the first *max_assets* columns after filtering.

        Returns
        -------
        pd.DataFrame
            Wide price frame indexed by date, columns are tickers.
        """
        effective_max = max_assets if max_assets is not None else self._max_assets

        engine = sa.create_engine(self._db_url)
        df_raw = self._query_raw(engine)

        ticker_to_exchange = self._build_ticker_exchange_map(df_raw)
        prices = self._pivot(df_raw)

        prices = self._calendar_ffill(prices, ticker_to_exchange)
        prices = self._filter_by_history(prices, df_raw)

        if effective_max is not None:
            prices = prices.iloc[:, :effective_max]

        total_nan_pct = prices.isna().mean().mean() * 100
        print(
            f"  Loaded {prices.shape[1]} tickers x {prices.shape[0]} days "
            f"({prices.index.min().date()} -> {prices.index.max().date()}) "
            f"| residual NaN after calendar-fill: {total_nan_pct:.2f}%"
        )
        return prices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_raw(engine: sa.Engine) -> pd.DataFrame:
        """Query raw price history joined with instrument and exchange data."""
        query = """
            SELECT i.ticker, e.name AS exchange_name, ph.date, ph.close
            FROM price_history ph
            JOIN instruments i ON ph.instrument_id = i.id
            JOIN exchanges e ON i.exchange_id = e.id
            WHERE i.is_active = true
              AND ph.close IS NOT NULL
              AND ph.close > 0
        """
        return pd.read_sql(query, engine, parse_dates=["date"])

    @staticmethod
    def _build_ticker_exchange_map(df_raw: pd.DataFrame) -> dict[str, str]:
        """Build a mapping of ticker -> exchange name from raw query results."""
        return (
            df_raw[["ticker", "exchange_name"]]
            .drop_duplicates("ticker")
            .set_index("ticker")["exchange_name"]
            .to_dict()
        )

    @staticmethod
    def _pivot(df_raw: pd.DataFrame) -> pd.DataFrame:
        """Pivot raw long-format data to wide price matrix."""
        prices = df_raw.pivot_table(index="date", columns="ticker", values="close")
        prices.sort_index(inplace=True)
        return prices

    @staticmethod
    def _build_non_trading_masks(
        all_dates: pd.DatetimeIndex,
        ticker_to_exchange: dict[str, str],
    ) -> dict[str, pd.Index]:
        """Return per-exchange sets of dates that are NOT trading days.

        Parameters
        ----------
        all_dates:
            Full date index of the pivot table.
        ticker_to_exchange:
            Mapping of ticker -> exchange name.

        Returns
        -------
        dict[str, pd.Index]
            Keys are exchange names; values are dates not in the exchange
            trading calendar within the data range.
        """
        start_str = all_dates.min().date().isoformat()
        end_str = all_dates.max().date().isoformat()

        non_trading: dict[str, pd.Index] = {}
        seen_exchanges: set[str] = set()

        for exchange_name, mic in EXCHANGE_NAME_TO_MIC.items():
            if exchange_name not in set(ticker_to_exchange.values()):
                continue
            if mic in seen_exchanges:
                continue
            seen_exchanges.add(mic)
            try:
                cal = xcals.get_calendar(mic)
                sessions = cal.sessions_in_range(start_str, end_str)
                trading_days = pd.DatetimeIndex(sessions).normalize()
                non_trading[exchange_name] = all_dates.difference(trading_days)
            except Exception as exc:
                print(
                    f"  Warning: could not load calendar for "
                    f"{exchange_name} ({mic}): {exc}"
                )

        return non_trading

    def _calendar_ffill(
        self,
        prices: pd.DataFrame,
        ticker_to_exchange: dict[str, str],
    ) -> pd.DataFrame:
        """Forward-fill only on non-trading-day gaps per exchange.

        Parameters
        ----------
        prices:
            Wide price pivot table before gap-filling.
        ticker_to_exchange:
            Mapping of ticker -> exchange name.

        Returns
        -------
        pd.DataFrame
            Price frame with calendar gaps filled, genuine missing data
            left as NaN.
        """
        all_dates = prices.index
        non_trading_by_exchange = self._build_non_trading_masks(
            all_dates, ticker_to_exchange
        )

        prices_filled = prices.copy()
        exchange_to_tickers: dict[str, list[str]] = {}
        for ticker, exch in ticker_to_exchange.items():
            if ticker in prices_filled.columns:
                exchange_to_tickers.setdefault(exch, []).append(ticker)

        for exchange_name, tickers in exchange_to_tickers.items():
            gap_dates = non_trading_by_exchange.get(exchange_name)
            if gap_dates is None or len(gap_dates) == 0:
                continue
            cols = [t for t in tickers if t in prices_filled.columns]
            prices_filled.loc[gap_dates, cols] = (
                prices_filled[cols].ffill().loc[gap_dates]
            )

        return prices_filled

    def _filter_by_history(
        self,
        prices: pd.DataFrame,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keep only tickers with enough genuine (pre-fill) trading days.

        Parameters
        ----------
        prices:
            Wide price frame after calendar-fill.
        df_raw:
            Original long-format query result used to count genuine rows.

        Returns
        -------
        pd.DataFrame
            Price frame restricted to tickers meeting the ``min_days``
            threshold.
        """
        genuine_counts = df_raw.groupby("ticker")["date"].count()
        good_tickers = genuine_counts[genuine_counts >= self._min_days].index
        good_tickers = good_tickers.intersection(prices.columns)
        return prices[good_tickers]
