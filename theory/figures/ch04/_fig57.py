"""Fig57CVaRVsVaR — two-panel histogram showing VaR vs CVaR under thin and fat tails."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from theory.figures._base import FigureGenerator

_N_SAMPLES = 10000
_RNG_SEED = 42
_ALPHA = 0.05


class Fig57CVaRVsVaR(FigureGenerator):
    """Two-panel: Normal vs Student-t(df=3) histograms with VaR/CVaR lines.

    Demonstrates that under thin tails VaR and CVaR are similar, but
    under fat tails CVaR is much more severe than VaR.
    """

    @property
    def name(self) -> str:
        return "fig_57_cvar_vs_var"

    def generate(self) -> None:
        rng = np.random.default_rng(_RNG_SEED)

        # Panel A: Normal (thin tails)
        normal_rets = rng.normal(0.0003, 0.012, _N_SAMPLES)

        # Panel B: Student-t (fat tails), scaled to similar volatility
        t_rets = stats.t.rvs(
            df=3, loc=0.0003, scale=0.008, size=_N_SAMPLES, random_state=rng,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for ax, rets, title, dist_label in [
            (ax1, normal_rets, "Panel A: Normal Distribution (Thin Tails)", "Normal"),
            (ax2, t_rets, "Panel B: Student-t (df=3, Fat Tails)", "Student-t"),
        ]:
            var = -np.quantile(rets, _ALPHA)
            tail = rets[rets <= -var]
            cvar = -np.mean(tail) if len(tail) > 0 else var
            gap = cvar - var

            print(
                f"  Fig 57 ({dist_label}): "
                f"VaR={var:.4f}, CVaR={cvar:.4f}, gap={gap:.4f}"
            )

            # Histogram
            ax.hist(
                rets, bins=100, density=True,
                color="#E0E0E0", edgecolor="#BDBDBD", linewidth=0.3,
            )

            # Shade the 5% tail
            threshold = np.quantile(rets, _ALPHA)
            bins_arr = np.linspace(rets.min(), threshold, 200)
            kde = stats.gaussian_kde(rets)
            ax.fill_between(
                bins_arr, kde(bins_arr), alpha=0.4, color="#F44336", label="5% tail",
            )

            # VaR line
            ax.axvline(
                -var, color="#E91E63", linewidth=2, linestyle="-",
                label=f"VaR 5% = {var:.2%}",
            )

            # CVaR line
            ax.axvline(
                -cvar, color="#9C27B0", linewidth=2, linestyle="--",
                label=f"CVaR 5% = {cvar:.2%}",
            )

            # Annotate gap
            ymax = ax.get_ylim()[1]
            ax.annotate(
                "", xy=(-cvar, ymax * 0.6), xytext=(-var, ymax * 0.6),
                arrowprops={"arrowstyle": "<->", "color": "#FF5722", "lw": 1.5},
            )
            ax.text(
                (-var - cvar) / 2, ymax * 0.65,
                f"Gap = {gap:.2%}",
                ha="center", fontsize=9, fontweight="bold", color="#FF5722",
            )

            ax.set_xlabel("Daily Return")
            ax.set_ylabel("Density")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8, loc="upper left")

        fig.suptitle(
            "VaR vs CVaR: Why Tail Shape Matters",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
