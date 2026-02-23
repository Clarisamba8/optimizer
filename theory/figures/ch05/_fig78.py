"""Fig78DRCVaR — DR-CVaR epsilon sweep showing conservatism-robustness tradeoff."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import DRCVaRConfig, build_dr_cvar
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 8


class Fig78DRCVaR(FigureGenerator):
    """DR-CVaR: portfolio CVaR and return vs Wasserstein ball radius epsilon.

    Sweeps epsilon from 0 to 0.10 and shows the conservatism tradeoff.
    """

    @property
    def name(self) -> str:
        return "fig_78_dr_cvar"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        epsilons = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
        portfolio_cvars = []
        portfolio_returns = []

        for eps in epsilons:
            cfg = DRCVaRConfig(epsilon=eps)
            opt = build_dr_cvar(cfg)
            try:
                opt.fit(ret)
                port_ret = ret.values @ opt.weights_
                cvar_val = -np.mean(
                    port_ret[port_ret <= np.quantile(port_ret, 0.05)]
                )
                portfolio_cvars.append(cvar_val * np.sqrt(252) * 100)
                portfolio_returns.append(np.mean(port_ret) * 252 * 100)
            except Exception as e:
                print(f"  Warning: epsilon={eps} failed: {e}")
                portfolio_cvars.append(np.nan)
                portfolio_returns.append(np.nan)

        fig, ax1 = plt.subplots(figsize=(11, 6))

        color_cvar = "#E91E63"
        color_ret = "#2196F3"

        ax1.plot(
            epsilons, portfolio_cvars, "o-",
            color=color_cvar, linewidth=2, markersize=8,
            label="CVaR 5% (annualized, %)",
        )
        ax1.set_xlabel(r"Wasserstein Ball Radius $\epsilon$")
        ax1.set_ylabel("Annualized CVaR 5% (%)", color=color_cvar)
        ax1.tick_params(axis="y", labelcolor=color_cvar)

        ax2 = ax1.twinx()
        ax2.plot(
            epsilons, portfolio_returns, "s--",
            color=color_ret, linewidth=2, markersize=8,
            label="Expected Return (annualized, %)",
        )
        ax2.set_ylabel("Annualized Return (%)", color=color_ret)
        ax2.tick_params(axis="y", labelcolor=color_ret)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="center right", fontsize=9, framealpha=0.9)

        ax1.set_title(
            "Distributionally Robust CVaR: Conservatism vs Robustness Tradeoff",
            fontsize=12, fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)

        # Mark epsilon=0 as baseline
        ax1.axvline(0, color="#9E9E9E", linestyle=":", alpha=0.5)
        ax1.text(0.001, ax1.get_ylim()[1] * 0.95, "Empirical\nCVaR",
                 fontsize=8, color="#9E9E9E", va="top")

        print(f"  Fig 78: DR-CVaR sweep over {len(epsilons)} epsilon values")

        plt.tight_layout()
        self._save(fig)
