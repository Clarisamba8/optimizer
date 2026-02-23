"""Fig63DiversificationScatter — random portfolios in (vol, DR) space."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.optimization import (
    MaxDiversificationConfig,
    MeanRiskConfig,
    build_max_diversification,
    build_mean_risk,
)
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_N_ASSETS = 15
_N_PORTFOLIOS = 500
_RNG_SEED = 42


class Fig63DiversificationScatter(FigureGenerator):
    """500 random Dirichlet portfolios in (vol, diversification ratio) space.

    Marks the MaxDiv and MinVar solutions to show the efficient frontier
    of diversification.
    """

    @property
    def name(self) -> str:
        return "fig_63_diversification_scatter"

    def generate(self) -> None:
        rng = np.random.default_rng(_RNG_SEED)

        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()

        top = returns.notna().sum().nlargest(_N_ASSETS).index
        ret = returns[top].dropna()

        cov = ret.cov().values
        vols = np.sqrt(np.diag(cov))

        # Random Dirichlet portfolios
        random_weights = rng.dirichlet(np.ones(_N_ASSETS), _N_PORTFOLIOS)

        port_vols = []
        port_drs = []
        for w in random_weights:
            pv = np.sqrt(w @ cov @ w)
            dr = (w @ vols) / pv
            port_vols.append(pv * np.sqrt(252) * 100)  # annualised %
            port_drs.append(dr)

        port_vols = np.array(port_vols)
        port_drs = np.array(port_drs)

        # Optimal portfolios
        md = build_max_diversification(MaxDiversificationConfig())
        md.fit(ret)
        w_md = md.weights_
        md_vol = np.sqrt(w_md @ cov @ w_md) * np.sqrt(252) * 100
        md_dr = (w_md @ vols) / (np.sqrt(w_md @ cov @ w_md))

        mv = build_mean_risk(MeanRiskConfig.for_min_variance())
        mv.fit(ret)
        w_mv = mv.weights_
        mv_vol = np.sqrt(w_mv @ cov @ w_mv) * np.sqrt(252) * 100
        mv_dr = (w_mv @ vols) / (np.sqrt(w_mv @ cov @ w_mv))

        print(
            f"  Fig 63: {_N_PORTFOLIOS} random portfolios, "
            f"MaxDiv DR = {md_dr:.2f}, MinVar DR = {mv_dr:.2f}"
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.scatter(
            port_vols, port_drs, c="#E0E0E0", edgecolors="#BDBDBD",
            s=15, alpha=0.5, label="Random portfolios",
        )

        # MaxDiv
        ax.scatter(
            md_vol, md_dr, c="#FF5722", s=200, marker="*", zorder=5,
            edgecolors="#BF360C", linewidth=1.5,
            label=f"Max Diversification (DR = {md_dr:.2f})",
        )

        # MinVar
        ax.scatter(
            mv_vol, mv_dr, c="#2196F3", s=200, marker="D", zorder=5,
            edgecolors="#0D47A1", linewidth=1.5,
            label=f"Min Variance (DR = {mv_dr:.2f})",
        )

        ax.set_xlabel("Annualised Volatility (%)")
        ax.set_ylabel("Diversification Ratio")
        ax.set_title(
            "Diversification Ratio Landscape: Random Portfolios vs Optimal",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9)

        plt.tight_layout()
        self._save(fig)
