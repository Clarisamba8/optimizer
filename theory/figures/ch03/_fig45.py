"""Fig45PickMatrix — heatmap of the pick matrix P for 3 canonical view types."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_ASSETS = [f"A{i}" for i in range(1, 11)]
_N = len(_ASSETS)


class Fig45PickMatrix(FigureGenerator):
    """Annotated heatmap of the pick matrix P.

    Three views on a 10-asset universe:
    1. Absolute view on asset 3 (P[0,2] = 1)
    2. Relative view: asset 5 outperforms asset 8 (P[1,4] = +1, P[1,7] = -1)
    3. Basket view on assets 1-4 (P[2,0:4] = 0.25 each)

    Colour scheme: +1 green, -1 red, fractions yellow, 0 light grey.
    """

    @property
    def name(self) -> str:
        return "fig_45_pick_matrix"

    def generate(self) -> None:
        pick = np.zeros((3, _N))

        # View 1: Absolute — Asset 3 returns 10%
        pick[0, 2] = 1.0

        # View 2: Relative — Asset 5 outperforms Asset 8
        pick[1, 4] = 1.0
        pick[1, 7] = -1.0

        # View 3: Basket — Assets 1-4 average
        pick[2, 0:4] = 0.25

        q_vec = np.array([0.10, 0.03, 0.06])

        view_labels = [
            "V1: Absolute (A3 = 10%)",
            "V2: Relative (A5 - A8 = 3%)",
            "V3: Basket (avg A1-A4 = 6%)",
        ]

        fig, ax = plt.subplots(figsize=(12, 4))

        # Custom coloring: map values to colors
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

        im = ax.imshow(pick, cmap=cmap, norm=norm, aspect="auto")

        # Annotate each cell
        for i in range(3):
            for j in range(_N):
                val = pick[i, j]
                if val == 0:
                    text = "0"
                    color = "#999"
                elif val == 1.0:
                    text = "+1"
                    color = "white"
                elif val == -1.0:
                    text = "-1"
                    color = "white"
                else:
                    text = f"{val:.2f}"
                    color = "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        ax.set_xticks(range(_N))
        ax.set_xticklabels(_ASSETS, fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(view_labels, fontsize=9)
        ax.set_xlabel("Assets")
        ax.set_title(
            r"Pick Matrix $\mathbf{P}$ Encodes View Structure"
            "\n3 canonical view types on a 10-asset universe",
            fontsize=11, fontweight="bold",
        )

        # Add Q column annotation on the right
        for i, q_val in enumerate(q_vec):
            ax.text(
                _N + 0.3, i, f"Q = {q_val:.0%}",
                ha="left", va="center", fontsize=10,
                fontweight="bold", color="#333",
            )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
        cbar.set_label("Pick weight", fontsize=9)

        plt.tight_layout()
        self._save(fig)
