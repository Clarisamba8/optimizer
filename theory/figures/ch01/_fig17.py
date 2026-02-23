"""Fig17ArithmeticVsLog — dual-panel arithmetic vs log return comparison."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theory.figures._base import FigureGenerator
from theory.figures._helpers import prices_to_log_returns, prices_to_returns


class Fig17ArithmeticVsLog(FigureGenerator):
    """Dual-panel: arithmetic vs log returns for 3 real assets + divergence.

    Stocks chosen for stable, large-cap characteristics (price ratio 1.4-1.6x)
    to cleanly illustrate the arithmetic vs log compounding concept without
    meme-stock or biotech noise distorting Panel B.
    """

    @property
    def name(self) -> str:
        return "fig_17_arithmetic_vs_log"

    def generate(self) -> None:
        prices = self._prices

        # Handpicked stable large-caps: Union Pacific, Sysco, Kimberly-Clark
        preferred = ["UNP_US_EQ", "SYY_US_EQ", "KMB_US_EQ"]
        available = [t for t in preferred if t in prices.columns]
        if len(available) < 3:
            log_rets = prices_to_log_returns(prices).dropna(
                axis=1, thresh=int(len(prices) * 0.95)
            )
            ann_vol = log_rets.std() * np.sqrt(252)
            fallback = (
                ann_vol[ann_vol.between(0.10, 0.30)]
                .nsmallest(3 - len(available))
                .index.tolist()
            )
            available = available + fallback
        tickers_3 = available[:3]
        p3 = prices[tickers_3].dropna()

        # Use most recent 252 trading days (approx 1 year)
        p3 = p3.tail(252)
        short_names = [t.replace("_US_EQ", "").replace("p_EQ", "") for t in tickers_3]

        # Normalise to 1 at start
        p3_norm = p3 / p3.iloc[0]

        arith_returns = p3_norm.pct_change().dropna()
        log_r = np.log(p3_norm / p3_norm.shift(1)).dropna()

        # Cross-sectional portfolio return error at each period (equal weights)
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        true_port = arith_returns.values @ weights
        log_port = np.expm1(log_r.values @ weights)
        divergence = np.abs(true_port - log_port) * 10_000  # bps

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        days = np.arange(len(arith_returns))

        for i, (ticker, name) in enumerate(zip(tickers_3, short_names)):
            cum_arith = (1 + arith_returns[ticker]).cumprod() - 1
            cum_log = np.exp(log_r[ticker].cumsum()) - 1
            ax1.plot(days, cum_arith * 100, color=colors[i], lw=1.5, label=f"{name} (arith)")
            ax1.plot(days, cum_log * 100, color=colors[i], lw=1.5, ls="--", alpha=0.6)

        legend_handles = [
            mpatches.Patch(color=c, label=n) for c, n in zip(colors, short_names)
        ] + [
            mpatches.Patch(color="grey", label="Solid = arithmetic"),
            mpatches.Patch(color="grey", fill=False, label="Dashed = log"),
        ]
        ax1.legend(handles=legend_handles, fontsize=8)
        ax1.set_xlabel("Trading Day")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.set_title("Panel A: Cumulative Wealth Paths\n(arithmetic vs log compounding)")
        ax1.axhline(0, color="black", lw=0.5, ls=":")

        ax2.fill_between(days, divergence, alpha=0.4, color="#E91E63")
        ax2.plot(days, divergence, color="#E91E63", lw=1.2)
        ax2.set_xlabel("Trading Day")
        ax2.set_ylabel("|Arithmetic - Log| Portfolio Return (bps)")
        ax2.set_title(
            "Panel B: Cross-Sectional Aggregation Error\n"
            "(log returns used incorrectly in MVO)"
        )

        fig.suptitle(
            f"Arithmetic vs Logarithmic Returns: Compounding and Cross-Sectional Error\n"
            f"({', '.join(short_names)} - stable large-caps, last 252 trading days)",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        self._save(fig)
