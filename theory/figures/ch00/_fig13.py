"""Fig13TimeVaryingWeights — Tilt multiplier time series by regime."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.factors._config import (
    FactorGroupType,
    MacroRegime,
    RegimeTiltConfig,
)
from optimizer.factors._regime import get_regime_tilts
from theory.figures._base import FigureGenerator

_REGIME_COLORS = {
    MacroRegime.EXPANSION: "#C8E6C9",
    MacroRegime.SLOWDOWN: "#FFF9C4",
    MacroRegime.RECESSION: "#FFCDD2",
    MacroRegime.RECOVERY: "#BBDEFB",
}


def _heuristic_regime_series(prices: pd.DataFrame) -> pd.Series:
    """Derive a quarterly regime series from market return and volatility.

    Uses rolling 12-month market return and 3-month volatility to
    assign a regime label at quarterly intervals.

    Returns
    -------
    pd.Series
        Quarterly regime labels indexed by date.
    """
    market_ret = prices.pct_change().mean(axis=1)
    quarterly_dates = market_ret.resample("QE").last().index

    regimes: list[MacroRegime] = []
    for date in quarterly_dates:
        loc = market_ret.index.get_indexer([date], method="ffill")[0]
        if loc < 63:
            regimes.append(MacroRegime.EXPANSION)
            continue

        # Rolling 6-month return (more responsive than 12-month)
        start_6m = max(0, loc - 126)
        cum_ret = (1 + market_ret.iloc[start_6m:loc + 1]).prod() - 1

        # Rolling 3-month volatility
        start_3m = max(0, loc - 63)
        vol = market_ret.iloc[start_3m:loc + 1].std() * np.sqrt(252)

        # Median volatility for adaptive threshold
        start_full = max(0, loc - 252)
        med_vol = market_ret.iloc[start_full:loc + 1].std() * np.sqrt(252)

        # Classification using return + volatility signals
        high_vol = vol > med_vol * 1.1
        if cum_ret > 0.05 and not high_vol:
            regimes.append(MacroRegime.EXPANSION)
        elif cum_ret > 0 and high_vol:
            regimes.append(MacroRegime.SLOWDOWN)
        elif cum_ret <= -0.02:
            regimes.append(MacroRegime.RECESSION)
        else:
            regimes.append(MacroRegime.RECOVERY)

    return pd.Series(regimes, index=quarterly_dates)


_TILTED_GROUPS = [
    FactorGroupType.VALUE,
    FactorGroupType.PROFITABILITY,
    FactorGroupType.MOMENTUM,
    FactorGroupType.LOW_RISK,
]

_TILTED_COLORS = {
    FactorGroupType.VALUE: "#1565C0",
    FactorGroupType.PROFITABILITY: "#E91E63",
    FactorGroupType.MOMENTUM: "#FF9800",
    FactorGroupType.LOW_RISK: "#9C27B0",
}


def _regime_tilt_series(regime_series: pd.Series) -> pd.DataFrame:
    """Build a DataFrame of raw tilt multipliers over time.

    Returns a DataFrame indexed by date with columns for each tilted
    factor group, values are the multiplicative tilt (1.0 = no change).
    """
    config = RegimeTiltConfig()
    records: list[dict[str, float]] = []
    for regime in regime_series:
        tilts = get_regime_tilts(regime, config)
        row = {g.value: tilts.get(g, 1.0) for g in _TILTED_GROUPS}
        records.append(row)
    return pd.DataFrame(records, index=regime_series.index)


class Fig13TimeVaryingWeights(FigureGenerator):
    """Stacked area chart of factor group weights with regime shading.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        PostgreSQL connection string (used for regime data; falls back
        to heuristic classification from price data).
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
        return "fig_13_time_varying_weights"

    def generate(self) -> None:
        prices = self._prices.ffill()
        if len(prices) < 252:
            print("  Fig 13: too few dates, using fallback.")
            self._generate_fallback()
            return

        regime_series = _heuristic_regime_series(prices)
        if len(regime_series) < 4:
            print("  Fig 13: too few quarterly dates, using fallback.")
            self._generate_fallback()
            return

        n_distinct = regime_series.nunique()
        if n_distinct < 2:
            print(f"  Fig 13: only {n_distinct} regime(s), using fallback.")
            self._generate_fallback()
            return

        tilt_df = _regime_tilt_series(regime_series)
        self._plot(tilt_df, regime_series)

    def _generate_fallback(self) -> None:
        """Synthetic regime-varying tilts for illustration."""
        n_quarters = 60
        dates = pd.date_range("2010-01-01", periods=n_quarters, freq="QE")
        regimes_cycle = [
            MacroRegime.EXPANSION, MacroRegime.EXPANSION,
            MacroRegime.SLOWDOWN, MacroRegime.RECESSION,
            MacroRegime.RECOVERY,
        ]
        regime_list = [
            regimes_cycle[i % len(regimes_cycle)]
            for i in range(n_quarters)
        ]
        regime_series = pd.Series(regime_list, index=dates)
        tilt_df = _regime_tilt_series(regime_series)
        self._plot(tilt_df, regime_series)

    def _plot(
        self,
        tilt_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> None:
        from matplotlib.patches import Patch

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(13, 7), height_ratios=[4, 1],
            sharex=True, gridspec_kw={"hspace": 0.08},
        )

        # --- Top panel: tilt multiplier step chart ---
        for group in _TILTED_GROUPS:
            col = group.value
            if col not in tilt_df.columns:
                continue
            label = col.replace("_", " ").title()
            ax_top.step(
                tilt_df.index, tilt_df[col].values,
                lw=2.2, color=_TILTED_COLORS[group],
                label=label, where="post",
            )

        # Regime background shading
        prev_date = tilt_df.index[0]
        for i in range(1, len(regime_series)):
            date = regime_series.index[i]
            regime = regime_series.iloc[i - 1]
            color = _REGIME_COLORS.get(regime, "#FFFFFF")
            ax_top.axvspan(prev_date, date, alpha=0.15, color=color, zorder=0)
            prev_date = date

        ax_top.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax_top.set_ylabel("Tilt Multiplier")
        ax_top.set_title(
            "Regime-Conditional Factor Weight Tilts Over Time",
        )
        ax_top.legend(fontsize=8, loc="upper right", ncol=2)
        ax_top.set_ylim(0.35, 1.65)

        # --- Bottom panel: regime bar chart ---
        regime_fill = {
            MacroRegime.EXPANSION: "#4CAF50",
            MacroRegime.SLOWDOWN: "#FFC107",
            MacroRegime.RECESSION: "#E91E63",
            MacroRegime.RECOVERY: "#2196F3",
        }

        prev_date = regime_series.index[0]
        for i in range(1, len(regime_series)):
            date = regime_series.index[i]
            regime = regime_series.iloc[i - 1]
            ax_bot.axvspan(
                prev_date, date,
                color=regime_fill.get(regime, "#9E9E9E"), alpha=0.8,
            )
            prev_date = date

        regime_patches = [
            Patch(facecolor=regime_fill[r], alpha=0.8, label=r.value.capitalize())
            for r in MacroRegime
        ]
        ax_bot.legend(
            handles=regime_patches, fontsize=7,
            loc="center left", ncol=4,
        )
        ax_bot.set_yticks([])
        ax_bot.set_ylabel("Regime", fontsize=9)
        ax_bot.set_xlabel("Date")

        fig.autofmt_xdate()
        plt.tight_layout()
        self._save(fig)
