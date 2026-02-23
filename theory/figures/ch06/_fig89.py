"""Fig89PerformanceRadar — radar chart for multiple strategies on multiple metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import (
    HRPConfig,
    MeanRiskConfig,
    build_equal_weighted,
    build_hrp,
    build_mean_risk,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8


class Fig89PerformanceRadar(FigureGenerator):
    """Radar/spider chart: 3 strategies on 6 performance metrics.

    Shows that strategy rankings differ across metrics.
    """

    @property
    def name(self) -> str:
        return "fig_89_performance_radar"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        strategies = {
            "Equal Weight": build_equal_weighted(),
            "Min Variance": build_mean_risk(MeanRiskConfig.for_min_variance()),
            "HRP": build_hrp(HRPConfig.for_variance()),
        }
        colors = ["#2196F3", "#E91E63", "#4CAF50"]

        metrics_data = {}
        for name, opt in strategies.items():
            try:
                opt.fit(ret)
                port_ret = ret.values @ opt.weights_

                ann_ret = np.mean(port_ret) * 252
                ann_vol = np.std(port_ret) * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

                downside = port_ret[port_ret < 0]
                downside_dev = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
                sortino = ann_ret / downside_dev if downside_dev > 0 else 0

                cum = np.cumprod(1 + port_ret)
                max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
                calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

                # CVaR ratio
                var_95 = -np.quantile(port_ret, 0.05)
                tail = port_ret[port_ret <= -var_95]
                cvar = -np.mean(tail) * np.sqrt(252) if len(tail) > 0 else var_95
                cvar_ratio = ann_ret / cvar if cvar > 0 else 0

                # Diversification (1 / HHI of weights)
                hhi = np.sum(opt.weights_**2)
                div_score = 1.0 / hhi if hhi > 0 else 0

                metrics_data[name] = [
                    sharpe, sortino, calmar, cvar_ratio, div_score,
                    1 - ann_vol,  # "stability" = 1 - vol
                ]
            except Exception as e:
                print(f"  Warning: {name} failed: {e}")

        if not metrics_data:
            print("  Fig 89: all strategies failed, skipping")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center")
            self._save(fig)
            return

        metric_names = ["Sharpe", "Sortino", "Calmar", "CVaR Ratio",
                        "Diversification", "Stability"]
        n_metrics = len(metric_names)

        # Normalize each metric to [0, 1] across strategies
        all_values = np.array(list(metrics_data.values()))
        mins = all_values.min(axis=0)
        maxs = all_values.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1

        normalized = {}
        for name, vals in metrics_data.items():
            normalized[name] = (np.array(vals) - mins) / ranges

        # Radar chart
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})

        for (name, vals), color in zip(normalized.items(), colors, strict=True):
            values = vals.tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", color=color, linewidth=2,
                    label=name, markersize=6)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

        ax.set_title(
            "Performance Ratio Radar: Different Metrics, Different Winners",
            fontsize=12, fontweight="bold", pad=20,
        )

        n_strat = len(metrics_data)
        print(f"  Fig 89: radar chart, {n_strat} strategies x {n_metrics} metrics")

        plt.tight_layout()
        self._save(fig)
