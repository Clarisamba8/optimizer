"""Fig82StrategyComparison — 5-strategy cumulative return comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.optimization import (
    HRPConfig,
    MeanRiskConfig,
    build_equal_weighted,
    build_hrp,
    build_inverse_volatility,
    build_mean_risk,
)
from optimizer.validation import WalkForwardConfig, build_walk_forward, run_cross_val
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8


class Fig82StrategyComparison(FigureGenerator):
    """Out-of-sample cumulative wealth for 5 strategies with walk-forward CV.

    Equal weight, inverse vol, min variance, max Sharpe, HRP.
    """

    @property
    def name(self) -> str:
        return "fig_82_strategy_comparison"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        strategies = {
            "Equal Weight": build_equal_weighted(),
            "Inverse Vol": build_inverse_volatility(),
            "Min Variance": build_mean_risk(MeanRiskConfig.for_min_variance()),
            "Max Sharpe": build_mean_risk(MeanRiskConfig.for_max_sharpe()),
            "HRP": build_hrp(HRPConfig.for_variance()),
        }
        colors = ["#9E9E9E", "#FF9800", "#2196F3", "#E91E63", "#4CAF50"]

        cv = build_walk_forward(WalkForwardConfig.for_quarterly_rolling())

        fig, ax = plt.subplots(figsize=(12, 7))

        stats_rows = []

        for (name, opt), color in zip(
            strategies.items(), colors, strict=True
        ):
            try:
                result = run_cross_val(opt, ret, cv=cv)
                port_returns = pd.Series(result.returns)

                cum = (1 + port_returns).cumprod()
                ax.plot(cum.values, color=color, linewidth=1.8, label=name)

                ann_ret = port_returns.mean() * 252 * 100
                ann_vol = port_returns.std() * np.sqrt(252) * 100
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                cum_vals = (1 + port_returns).cumprod()
                max_dd = ((cum_vals / cum_vals.cummax()) - 1).min() * 100

                stats_rows.append({
                    "name": name, "ret": ann_ret, "vol": ann_vol,
                    "sharpe": sharpe, "max_dd": max_dd,
                })
            except Exception as e:
                print(f"  Warning: {name} failed: {e}")

        ax.set_xlabel("Trading Days (out-of-sample)")
        ax.set_ylabel("Cumulative Wealth ($1 invested)")
        ax.set_title(
            "Naive vs Sophisticated Strategies: Out-of-Sample Performance",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Summary table as text box
        if stats_rows:
            header = f"{'Strategy':<15} {'Ret%':>6} {'Vol%':>6} {'SR':>5} {'MDD%':>6}"
            lines = [header, "-" * 42]
            for s in stats_rows:
                lines.append(
                    f"{s['name']:<15} {s['ret']:>5.1f} {s['vol']:>5.1f} "
                    f"{s['sharpe']:>5.2f} {s['max_dd']:>5.1f}"
                )
            table_text = "\n".join(lines)
            ax.text(
                0.02, 0.35, table_text, transform=ax.transAxes,
                fontsize=7, fontfamily="monospace", va="top",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#BDBDBD"},
            )

        print(f"  Fig 82: {len(stats_rows)} strategies, {_N_ASSETS} assets")

        plt.tight_layout()
        self._save(fig)
