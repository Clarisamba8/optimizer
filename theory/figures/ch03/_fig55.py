"""Fig55SentimentMatrix — 3x3 sentiment-fundamental interaction heatmap."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from theory.figures._base import FigureGenerator


class Fig55SentimentMatrix(FigureGenerator):
    """Static 3x3 heatmap: sentiment-fundamental interaction matrix.

    Rows: fundamental view direction (Bullish / Neutral / Bearish)
    Columns: sentiment signal (Positive / Mixed / Negative)
    Cells: resulting confidence adjustment with color coding.

    Pure matplotlib — no optimizer modules needed.
    """

    @property
    def name(self) -> str:
        return "fig_55_sentiment_matrix"

    def generate(self) -> None:
        # Confidence adjustment values (positive = increase, negative = decrease)
        matrix = np.array([
            [+0.25, +0.05, -0.15],   # Bullish fundamental
            [+0.10,  0.00, -0.10],   # Neutral fundamental
            [-0.15, -0.05, +0.25],   # Bearish fundamental
        ])

        labels = np.array([
            ["Strong\nIncrease\n(+25%)", "Slight\nIncrease\n(+5%)",
             "Moderate\nDecrease\n(-15%)"],
            ["Slight\nIncrease\n(+10%)", "No\nChange\n(0%)",
             "Slight\nDecrease\n(-10%)"],
            ["Moderate\nDecrease\n(-15%)", "Slight\nDecrease\n(-5%)",
             "Strong\nIncrease\n(+25%)"],
        ])

        row_labels = [
            "Bullish\nFundamental", "Neutral\nFundamental",
            "Bearish\nFundamental",
        ]
        col_labels = ["Positive\nSentiment", "Mixed\nSentiment", "Negative\nSentiment"]

        fig, ax = plt.subplots(figsize=(9, 7))

        norm = TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25)
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        # Annotate cells
        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                text_color = "white" if abs(val) > 0.15 else "black"
                ax.text(
                    j, i, labels[i, j],
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color,
                )

        ax.set_xticks(range(3))
        ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
        ax.set_yticks(range(3))
        ax.set_yticklabels(row_labels, fontsize=10, fontweight="bold")

        ax.set_xlabel("News Sentiment Signal", fontsize=11, labelpad=10)
        ax.set_ylabel("Fundamental View Direction", fontsize=11, labelpad=10)

        ax.set_title(
            "Sentiment-Fundamental Interaction Matrix for View Confidence\n"
            "How news sentiment modulates the precision of active views",
            fontsize=11, fontweight="bold", pad=15,
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
        cbar.set_label(r"Confidence adjustment $\Delta\alpha_k$", fontsize=9)

        # Add explanatory note
        ax.text(
            0.5, -0.18,
            "Diagonal: Reinforcement (sentiment aligns with fundamental view)\n"
            "Off-diagonal: Contradiction (sentiment opposes fundamental view)",
            transform=ax.transAxes, fontsize=9, ha="center",
            style="italic", color="#555",
        )

        plt.tight_layout()
        self._save(fig)
