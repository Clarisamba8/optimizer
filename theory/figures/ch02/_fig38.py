"""Fig38FactorModelParams — free parameters: full covariance vs factor model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_N_VALUES = [50, 100, 200, 500]
_K_VALUES = [3, 5, 10]
_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]


class Fig38FactorModelParams(FigureGenerator):
    """Bar chart comparing free parameters in covariance estimation.

    Three representations are shown:
    - Full covariance: N*(N+1)/2  — grows quadratically in N
    - Factor model K*N + N = N*(K+1) — grows linearly in N for fixed K
    - Ledoit-Wolf: same count as full covariance but shown as dotted overlay
      to emphasise that regularisation reduces *effective* dimensionality
      without reducing the raw parameter count.

    No price data is required; all values are derived from pure arithmetic.
    """

    @property
    def name(self) -> str:
        return "fig_38_factor_model_params"

    def generate(self) -> None:  # noqa: PLR0914
        n_arr = np.array(_N_VALUES, dtype=float)

        full_params = n_arr * (n_arr + 1) / 2

        fig, ax = plt.subplots(figsize=(10, 6))

        bar_width = 0.18
        x = np.arange(len(_N_VALUES))

        # Full covariance bars (grey)
        ax.bar(
            x - bar_width * (len(_K_VALUES) / 2),
            full_params,
            width=bar_width,
            label="Full covariance  N(N+1)/2",
            color="#9E9E9E",
            edgecolor="white",
            linewidth=0.5,
        )

        # Factor model bars, one per K value
        for idx, (k, color) in enumerate(zip(_K_VALUES, _COLORS)):
            factor_params = n_arr * k + n_arr  # K*N (loadings) + N (specific vars)
            offset = bar_width * (idx - len(_K_VALUES) / 2 + 1)
            ax.bar(
                x + offset,
                factor_params,
                width=bar_width,
                label=f"Factor model  K={k}:  N(K+1)",
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

        # Ledoit-Wolf dotted overlay (same count as full cov — highlight regularisation)
        ax.plot(
            x - bar_width * (len(_K_VALUES) / 2),
            full_params,
            color="#424242",
            linestyle=":",
            linewidth=1.8,
            marker="D",
            markersize=5,
            label="Ledoit-Wolf (same count, regularised)",
            zorder=5,
        )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N = {n}" for n in _N_VALUES])
        ax.set_xlabel("Universe size  N")
        ax.set_ylabel("Number of free parameters  (log scale)")
        ax.set_title(
            "Factor Model Dimensionality Reduction: Parameters vs Universe Size\n"
            "Factor model grows linearly vs quadratic full covariance",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Annotate reduction at N=500
        n500 = 500.0
        full_500 = n500 * (n500 + 1) / 2
        factor_500_k5 = n500 * (_K_VALUES[1] + 1)
        reduction_pct = (1 - factor_500_k5 / full_500) * 100
        ax.annotate(
            f"K=5: {reduction_pct:.0f}% reduction\nat N=500",
            xy=(x[-1] + bar_width * 0.5, factor_500_k5),
            xytext=(x[-1] - 0.8, full_500 * 0.3),
            arrowprops=dict(arrowstyle="->", color="#333"),
            fontsize=9,
            color="#333",
        )

        plt.tight_layout()
        self._save(fig)
