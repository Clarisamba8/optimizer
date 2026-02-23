"""Fig81BenchmarkTracking — cumulative returns + rolling tracking error."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.optimization import BenchmarkTrackerConfig, build_benchmark_tracker
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 10


class Fig81BenchmarkTracking(FigureGenerator):
    """Dual-panel: cumulative returns of tracker vs benchmark,
    and rolling 12-month tracking error.
    """

    @property
    def name(self) -> str:
        return "fig_81_benchmark_tracking"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS + 1).index
        # Use first asset as benchmark proxy, rest as universe
        benchmark_col = top[0]
        universe_cols = top[1: _N_ASSETS + 1]

        ret_universe = returns[universe_cols].dropna()
        ret_benchmark = returns[benchmark_col].reindex(ret_universe.index).dropna()
        ret_universe = ret_universe.reindex(ret_benchmark.index)

        # Fit benchmark tracker
        cfg = BenchmarkTrackerConfig()
        tracker = build_benchmark_tracker(cfg)
        tracker.fit(ret_universe, ret_benchmark)

        # Portfolio returns
        port_ret = ret_universe.values @ tracker.weights_
        port_ret_series = pd.Series(port_ret, index=ret_universe.index)

        # Cumulative returns
        cum_port = (1 + port_ret_series).cumprod()
        cum_bench = (1 + ret_benchmark).cumprod()

        # Rolling tracking error (252-day window)
        te_window = 252
        active_ret = port_ret_series - ret_benchmark
        rolling_te = active_ret.rolling(te_window).std() * np.sqrt(252) * 100

        bench_label = benchmark_col.replace("_US_EQ", "").replace("_EQ", "")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        height_ratios=[2, 1], sharex=True)

        # Panel A: Cumulative returns
        ax1.plot(cum_port.index, cum_port.values, color="#2196F3",
                 linewidth=1.5, label="Tracking Portfolio")
        ax1.plot(cum_bench.index, cum_bench.values, color="#E91E63",
                 linewidth=1.5, label=f"Benchmark ({bench_label})")
        ax1.fill_between(
            cum_port.index,
            cum_port.values,
            cum_bench.values,
            alpha=0.1, color="#FF9800",
        )
        ax1.set_ylabel("Cumulative Return")
        ax1.set_title(
            "Benchmark Tracking: Portfolio vs Index with Tracking Error Band",
            fontsize=12, fontweight="bold",
        )
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Panel B: Rolling TE
        valid_te = rolling_te.dropna()
        ax2.plot(valid_te.index, valid_te.values, color="#FF9800", linewidth=1.5)
        ax2.fill_between(valid_te.index, 0, valid_te.values,
                         color="#FF9800", alpha=0.2)
        ax2.set_ylabel("Tracking Error\n(ann. %)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        if len(valid_te) > 0:
            mean_te = valid_te.mean()
            ax2.axhline(mean_te, color="#E91E63", linestyle="--",
                        linewidth=1, label=f"Mean TE = {mean_te:.2f}%")
            ax2.legend(fontsize=8)

        print(
            f"  Fig 81: tracker vs {bench_label}, "
            f"{len(ret_universe)} days, {_N_ASSETS} assets"
        )

        plt.tight_layout()
        self._save(fig)
