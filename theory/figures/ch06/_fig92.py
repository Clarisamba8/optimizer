"""Fig92RebalancingFrequency — rebalancing frequency tradeoff bar chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.rebalancing import (
    TRADING_DAYS,
    RebalancingFrequency,
    compute_drifted_weights,
    compute_rebalancing_cost,
    compute_turnover,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8
_TC_BPS = 15  # 15 bps per trade


class Fig92RebalancingFrequency(FigureGenerator):
    """Grouped bar chart: turnover, gross return, costs, net return
    for each rebalancing frequency.
    """

    @property
    def name(self) -> str:
        return "fig_92_rebalancing_frequency"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        # Fit optimizer once for target weights
        opt = build_mean_risk(MeanRiskConfig.for_min_variance())
        opt.fit(ret)
        target_w = opt.weights_

        tc = _TC_BPS / 10_000  # Convert bps to decimal
        ret_vals = ret.values

        frequencies = [
            RebalancingFrequency.MONTHLY,
            RebalancingFrequency.QUARTERLY,
            RebalancingFrequency.SEMIANNUAL,
            RebalancingFrequency.ANNUAL,
        ]
        freq_labels = ["Monthly", "Quarterly", "Semi-Annual", "Annual"]

        results = {}
        for freq, label in zip(frequencies, freq_labels, strict=True):
            period = TRADING_DAYS[freq]
            current_w = target_w.copy()
            total_turnover = 0.0
            total_cost = 0.0
            portfolio_values = [1.0]

            for t in range(len(ret_vals)):
                daily_ret = ret_vals[t]
                port_ret = np.dot(current_w, daily_ret)
                current_w = compute_drifted_weights(current_w, daily_ret)

                if (t + 1) % period == 0:
                    turnover = compute_turnover(current_w, target_w)
                    cost = compute_rebalancing_cost(current_w, target_w, tc)
                    total_turnover += turnover
                    total_cost += cost
                    current_w = target_w.copy()
                    port_ret -= cost

                portfolio_values.append(portfolio_values[-1] * (1 + port_ret))

            n_years = len(ret_vals) / 252
            ann_turnover = total_turnover / n_years * 100
            ann_cost = total_cost / n_years * 100
            gross_ret = (portfolio_values[-1] ** (1 / n_years) - 1) * 100
            net_ret = gross_ret - ann_cost

            results[label] = {
                "turnover": ann_turnover,
                "gross_ret": gross_ret,
                "cost": ann_cost,
                "net_ret": net_ret,
            }

        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(freq_labels))
        width = 0.2

        metrics = [
            ("turnover", "Ann. Turnover (%)", "#9E9E9E"),
            ("gross_ret", "Gross Return (%)", "#2196F3"),
            ("cost", "Transaction Cost (%)", "#E91E63"),
            ("net_ret", "Net Return (%)", "#4CAF50"),
        ]

        for i, (key, label, color) in enumerate(metrics):
            values = [results[f][key] for f in freq_labels]
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=label,
                   color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(freq_labels, fontsize=10)
        ax.set_ylabel("Annualized (%)")
        ax.set_title(
            "Rebalancing Frequency Trade-Off: Gross Return vs Transaction Costs",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        print(f"  Fig 92: {len(frequencies)} frequencies, {_TC_BPS} bps cost")

        plt.tight_layout()
        self._save(fig)
