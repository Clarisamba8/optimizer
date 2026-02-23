"""Fig59RiskRadar — spider chart comparing risk measures across strategies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_RNG_SEED = 42


class Fig59RiskRadar(FigureGenerator):
    """Spider/radar chart: 7 risk measures x 5 strategies.

    Demonstrates that portfolio rankings change across risk measures —
    a portfolio that is best under variance may be worst under CVaR.
    """

    @property
    def name(self) -> str:
        return "fig_59_risk_radar"

    def generate(self) -> None:
        # Risk measure labels
        measures = [
            "Std Dev", "Semi-Dev", "CVaR 5%",
            "EVaR 5%", "Max DD", "Ulcer Idx", "MAD",
        ]
        n_measures = len(measures)

        # Strategy labels
        strategies = [
            "Min Variance", "Max Sharpe", "Risk Parity",
            "Max Diversification", "Equal Weight",
        ]
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9E9E9E"]

        # Synthetic risk values (normalised 0-1): designed to show that rankings change
        # Each row is a strategy, each column is a risk measure
        raw_risks = np.array([
            [0.20, 0.22, 0.45, 0.55, 0.60, 0.35, 0.18],  # MinVar: low std, high tail
            [0.50, 0.40, 0.35, 0.40, 0.45, 0.30, 0.42],  # MaxSharpe: moderate
            [0.35, 0.30, 0.30, 0.35, 0.40, 0.25, 0.30],  # RiskParity: balanced
            [0.30, 0.28, 0.32, 0.38, 0.50, 0.28, 0.27],  # MaxDiv: good diversification
            [0.55, 0.50, 0.50, 0.60, 0.55, 0.45, 0.48],  # EqualWeight: generally higher
        ])

        # Normalise each measure to [0.1, 1.0] for better visual spread
        col_min = raw_risks.min(axis=0)
        col_max = raw_risks.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        normalised = 0.1 + 0.9 * (raw_risks - col_min) / col_range

        # Radar chart setup
        angles = np.linspace(0, 2 * np.pi, n_measures, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})

        for i, (strat, color) in enumerate(zip(strategies, colors, strict=True)):
            values = normalised[i].tolist()
            values += values[:1]
            ax.plot(
                angles, values, "o-",
                linewidth=2, color=color, label=strat, markersize=5,
            )
            ax.fill(angles, values, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(measures, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["Low", "", "", "High"], fontsize=8)
        ax.set_rlabel_position(30)

        ax.set_title(
            "Portfolio Rankings Depend on the Risk Measure Chosen\n"
            "(higher = more risk)",
            fontsize=12, fontweight="bold", pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        print(
            f"  Fig 59: radar chart for {len(strategies)} strategies"
            f" x {n_measures} measures"
        )

        plt.tight_layout()
        self._save(fig)
