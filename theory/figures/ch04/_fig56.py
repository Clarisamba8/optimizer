"""Fig56AnnotatedRiskMeasures — annotated return distribution with all risk measures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from theory.figures._base import FigureGenerator

_N_SAMPLES = 5000
_RNG_SEED = 42
_ALPHA = 0.05  # 5% confidence level


class Fig56AnnotatedRiskMeasures(FigureGenerator):
    """Skew-normal distribution with VaR/CVaR/semi-dev/std bands annotated.

    Shows a single slightly left-skewed return distribution with vertical
    lines marking all major risk measures discussed in the chapter.
    """

    @property
    def name(self) -> str:
        return "fig_56_annotated_risk_measures"

    def generate(self) -> None:
        rng = np.random.default_rng(_RNG_SEED)

        # Skew-normal with negative skew (left tail heavier)
        raw = stats.skewnorm.rvs(
            a=-3, loc=0.0005, scale=0.015, size=_N_SAMPLES, random_state=rng
        )

        mean = np.mean(raw)
        std = np.std(raw, ddof=1)
        semi_dev = np.sqrt(np.mean(np.minimum(raw - mean, 0) ** 2))
        var_5 = -np.quantile(raw, _ALPHA)
        tail = raw[raw <= -var_5]
        cvar_5 = -np.mean(tail) if len(tail) > 0 else var_5

        print(
            f"  Fig 56: mean={mean:.4f}, std={std:.4f}, semi-dev={semi_dev:.4f}, "
            f"VaR_5%={var_5:.4f}, CVaR_5%={cvar_5:.4f}"
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # Histogram
        ax.hist(
            raw, bins=80, density=True,
            color="#E0E0E0", edgecolor="#BDBDBD", linewidth=0.5,
        )

        # KDE overlay
        x_kde = np.linspace(raw.min(), raw.max(), 500)
        kde = stats.gaussian_kde(raw)
        ax.plot(x_kde, kde(x_kde), color="#212121", linewidth=1.5)

        # Shade the 5% left tail
        x_tail = x_kde[x_kde <= np.quantile(raw, _ALPHA)]
        if len(x_tail) > 0:
            ax.fill_between(
                x_tail, kde(x_tail), alpha=0.3, color="#F44336", label="5% tail",
            )

        # Annotate risk measures
        ymax = ax.get_ylim()[1]

        # Standard deviation bands
        ax.axvline(
            mean - std, color="#2196F3", linestyle="--",
            linewidth=1.2, alpha=0.7,
            label=r"$\pm 1\sigma, \pm 2\sigma$",
        )
        ax.axvline(
            mean + std, color="#2196F3", linestyle="--",
            linewidth=1.2, alpha=0.7,
        )
        ax.axvline(
            mean - 2 * std, color="#2196F3", linestyle=":",
            linewidth=1.2, alpha=0.7,
        )
        ax.axvline(
            mean + 2 * std, color="#2196F3", linestyle=":",
            linewidth=1.2, alpha=0.7,
        )

        # Mean
        ax.axvline(
            mean, color="#4CAF50", linestyle="-", linewidth=1.5,
            label=f"Mean = {mean:.4f}",
        )

        # Semi-deviation boundary
        ax.axvline(
            mean - semi_dev, color="#FF9800", linestyle="-.", linewidth=1.5,
            label=f"Semi-dev = {semi_dev:.4f}",
        )

        # VaR
        ax.axvline(
            -var_5, color="#E91E63", linestyle="-", linewidth=2,
            label=f"VaR 5% = {var_5:.4f}",
        )

        # CVaR
        ax.axvline(
            -cvar_5, color="#9C27B0", linestyle="-", linewidth=2,
            label=f"CVaR 5% = {cvar_5:.4f}",
        )

        # Annotations with arrows
        ax.annotate(
            f"VaR$_{{5\\%}}$ = {var_5:.2%}",
            xy=(-var_5, ymax * 0.15), xytext=(-var_5 - 0.01, ymax * 0.55),
            fontsize=9, fontweight="bold", color="#E91E63",
            arrowprops={"arrowstyle": "->", "color": "#E91E63", "lw": 1.2},
        )
        ax.annotate(
            f"CVaR$_{{5\\%}}$ = {cvar_5:.2%}",
            xy=(-cvar_5, ymax * 0.10), xytext=(-cvar_5 - 0.008, ymax * 0.40),
            fontsize=9, fontweight="bold", color="#9C27B0",
            arrowprops={"arrowstyle": "->", "color": "#9C27B0", "lw": 1.2},
        )

        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.set_title(
            "Risk Measures on a Single Return Distribution",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        plt.tight_layout()
        self._save(fig)
