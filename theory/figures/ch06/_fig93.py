"""Fig93TradeTriggers — calendar vs threshold trade trigger time series."""

from __future__ import annotations

import matplotlib.pyplot as plt

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.rebalancing import (
    ThresholdRebalancingConfig,
    compute_drifted_weights,
    compute_turnover,
    should_rebalance,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8
_CALENDAR_PERIOD = 63  # Quarterly


class Fig93TradeTriggers(FigureGenerator):
    """Dual-panel: calendar vs threshold rebalancing trade triggers.

    Shows regular calendar trades vs irregular threshold-based trades
    with cumulative turnover comparison.
    """

    @property
    def name(self) -> str:
        return "fig_93_trade_triggers"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        # Use last 2 years
        ret = ret.iloc[-504:]

        opt = build_mean_risk(MeanRiskConfig.for_min_variance())
        opt.fit(ret)
        target_w = opt.weights_

        ret_vals = ret.values
        dates = ret.index
        threshold_cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)

        # Simulate calendar rebalancing
        cal_trades = []
        cal_turnover = []
        cal_cum_turnover = 0.0
        w_cal = target_w.copy()

        for t in range(len(ret_vals)):
            w_cal = compute_drifted_weights(w_cal, ret_vals[t])
            if (t + 1) % _CALENDAR_PERIOD == 0:
                tv = compute_turnover(w_cal, target_w)
                cal_trades.append(t)
                cal_cum_turnover += tv
                w_cal = target_w.copy()
            cal_turnover.append(cal_cum_turnover)

        # Simulate threshold rebalancing
        thr_trades = []
        thr_turnover = []
        thr_cum_turnover = 0.0
        w_thr = target_w.copy()

        for t in range(len(ret_vals)):
            w_thr = compute_drifted_weights(w_thr, ret_vals[t])
            if should_rebalance(w_thr, target_w, threshold_cfg):
                tv = compute_turnover(w_thr, target_w)
                thr_trades.append(t)
                thr_cum_turnover += tv
                w_thr = target_w.copy()
            thr_turnover.append(thr_cum_turnover)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Panel A: Calendar triggers
        ax1.set_title("Calendar Rebalancing (Quarterly)", fontsize=11,
                       fontweight="bold")
        for t in cal_trades:
            ax1.axvline(dates[t], color="#2196F3", alpha=0.4, linewidth=1.5)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(dates, cal_turnover, color="#E91E63", linewidth=1.5,
                      label=f"Cum. Turnover = {cal_cum_turnover:.2f}")
        ax1_twin.set_ylabel("Cumulative Turnover", color="#E91E63")
        ax1.set_ylabel("Trade Triggers")
        ax1.set_yticks([])
        ax1_twin.legend(loc="upper left", fontsize=9)
        ax1.text(0.98, 0.95, f"{len(cal_trades)} trades",
                 transform=ax1.transAxes, ha="right", va="top", fontsize=10,
                 fontweight="bold", color="#2196F3")

        # Panel B: Threshold triggers
        ax2.set_title("Threshold Rebalancing (5% absolute drift)", fontsize=11,
                       fontweight="bold")
        for t in thr_trades:
            ax2.axvline(dates[t], color="#FF9800", alpha=0.4, linewidth=1.5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(dates, thr_turnover, color="#E91E63", linewidth=1.5,
                      label=f"Cum. Turnover = {thr_cum_turnover:.2f}")
        ax2_twin.set_ylabel("Cumulative Turnover", color="#E91E63")
        ax2.set_ylabel("Trade Triggers")
        ax2.set_yticks([])
        ax2.set_xlabel("Date")
        ax2_twin.legend(loc="upper left", fontsize=9)
        ax2.text(0.98, 0.95, f"{len(thr_trades)} trades",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=10,
                 fontweight="bold", color="#FF9800")

        fig.suptitle(
            "Calendar vs Threshold Rebalancing: Trade Trigger Patterns",
            fontsize=13, fontweight="bold",
        )

        print(
            f"  Fig 93: calendar={len(cal_trades)} trades, "
            f"threshold={len(thr_trades)} trades"
        )

        plt.tight_layout()
        self._save(fig)
