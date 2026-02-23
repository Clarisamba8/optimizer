"""Fig94GrossVsNet — gross vs net cumulative returns."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.rebalancing import (
    compute_drifted_weights,
    compute_rebalancing_cost,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8
_TC_BPS = 15  # 15 bps per trade
_REBAL_PERIOD = 21  # Monthly (high-turnover)


class Fig94GrossVsNet(FigureGenerator):
    """Gross vs net cumulative returns for a high-turnover strategy.

    Shows the growing gap between gross and net performance over time.
    """

    @property
    def name(self) -> str:
        return "fig_94_gross_vs_net"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        opt = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        opt.fit(ret)
        target_w = opt.weights_

        tc = _TC_BPS / 10_000
        ret_vals = ret.values
        dates = ret.index

        gross_values = [1.0]
        net_values = [1.0]
        current_w = target_w.copy()
        total_cost = 0.0

        for t in range(len(ret_vals)):
            daily_ret = ret_vals[t]
            port_ret_gross = np.dot(current_w, daily_ret)
            current_w = compute_drifted_weights(current_w, daily_ret)

            cost = 0.0
            if (t + 1) % _REBAL_PERIOD == 0:
                cost = compute_rebalancing_cost(current_w, target_w, tc)
                total_cost += cost
                current_w = target_w.copy()

            gross_values.append(gross_values[-1] * (1 + port_ret_gross))
            net_values.append(net_values[-1] * (1 + port_ret_gross - cost))

        gross_arr = np.array(gross_values[1:])
        net_arr = np.array(net_values[1:])

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(dates, gross_arr, color="#2196F3", linewidth=1.8,
                label="Gross Returns")
        ax.plot(dates, net_arr, color="#E91E63", linewidth=1.8,
                label="Net Returns")
        ax.fill_between(
            dates, gross_arr, net_arr,
            color="#FF9800", alpha=0.15, label="Cost Drag",
        )

        # Annotate final gap
        final_gap = (gross_arr[-1] - net_arr[-1]) / gross_arr[-1] * 100
        n_years = len(ret_vals) / 252

        ax.annotate(
            f"Total cost drag: {final_gap:.1f}%\n"
            f"({total_cost / n_years * 10_000:.0f} bps/year)",
            xy=(dates[-1], (gross_arr[-1] + net_arr[-1]) / 2),
            xytext=(dates[int(len(dates) * 0.65)],
                    float(gross_arr.max()) * 0.95),
            fontsize=10, fontweight="bold", color="#E91E63",
            arrowprops={
                "arrowstyle": "->", "color": "#E91E63", "lw": 1.5,
            },
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Wealth ($1 invested)")
        ax.set_title(
            "The Hidden Tax: Gross vs Net Performance for a High-Turnover Strategy",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        print(
            f"  Fig 94: gross vs net over {n_years:.1f} years, "
            f"total cost = {total_cost * 100:.2f}%"
        )

        plt.tight_layout()
        self._save(fig)
