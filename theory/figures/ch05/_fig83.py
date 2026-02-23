"""Fig83StackingSchematic — ensemble stacking block diagram."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from theory.figures._base import FigureGenerator


class Fig83StackingSchematic(FigureGenerator):
    """Block diagram of ensemble stacking with 3 base optimizers + meta-optimizer.

    Pure matplotlib.patches diagram — no data needed.
    """

    @property
    def name(self) -> str:
        return "fig_83_stacking_schematic"

    def generate(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Input: Return matrix
        input_box = mpatches.FancyBboxPatch(
            (0.3, 3.0), 2.0, 2.0,
            boxstyle="round,pad=0.15", facecolor="#F5F5F5",
            edgecolor="#616161", linewidth=1.5,
        )
        ax.add_patch(input_box)
        ax.text(1.3, 4.3, "Return Matrix", ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(1.3, 3.7, r"$\mathbf{X} \in \mathbb{R}^{T \times N}$",
                ha="center", va="center", fontsize=9)

        # Arrows from input to base optimizers
        base_y_positions = [6.0, 4.0, 2.0]
        base_names = [
            "MeanRisk\n(Max Sharpe)", "HRP\n(Variance)",
            "RiskBudgeting\n(ERC)",
        ]
        base_colors = ["#2196F3", "#4CAF50", "#FF9800"]

        for y_pos in base_y_positions:
            ax.annotate(
                "", xy=(3.3, y_pos), xytext=(2.4, 4.0),
                arrowprops={"arrowstyle": "-|>", "color": "#424242", "lw": 1.5},
            )

        # Base optimizers
        for _i, (y_pos, name, color) in enumerate(
            zip(base_y_positions, base_names, base_colors, strict=True)
        ):
            box = mpatches.FancyBboxPatch(
                (3.3, y_pos - 0.7), 2.8, 1.4,
                boxstyle="round,pad=0.15", facecolor=color,
                edgecolor=color, alpha=0.2, linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(4.7, y_pos, name, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

        ax.text(4.7, 7.3, "Stage 1: Base Optimizers",
                ha="center", fontsize=11, fontweight="bold", color="#424242")

        # Arrows from base to sub-portfolios
        sub_y_positions = base_y_positions
        for i, y_pos in enumerate(sub_y_positions):
            ax.annotate(
                "", xy=(6.8, y_pos), xytext=(6.1, y_pos),
                arrowprops={
                    "arrowstyle": "-|>", "color": base_colors[i], "lw": 1.5,
                },
            )

        # Sub-portfolio weight vectors
        sub_labels = [
            r"$\mathbf{w}_1$" + "\n[0.25, 0.35, ...]",
            r"$\mathbf{w}_2$" + "\n[0.18, 0.22, ...]",
            r"$\mathbf{w}_3$" + "\n[0.20, 0.20, ...]",
        ]
        for i, (y_pos, label) in enumerate(
            zip(sub_y_positions, sub_labels, strict=True)
        ):
            box = mpatches.FancyBboxPatch(
                (6.8, y_pos - 0.5), 2.0, 1.0,
                boxstyle="round,pad=0.1", facecolor=base_colors[i],
                edgecolor=base_colors[i], alpha=0.15, linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(7.8, y_pos, label, ha="center", va="center",
                    fontsize=8, color=base_colors[i])

        ax.text(7.8, 7.3, "Sub-Portfolios",
                ha="center", fontsize=11, fontweight="bold", color="#424242")

        # Arrows from sub-portfolios to meta-optimizer
        for y_pos in sub_y_positions:
            ax.annotate(
                "", xy=(9.5, 4.0), xytext=(8.8, y_pos),
                arrowprops={"arrowstyle": "-|>", "color": "#424242", "lw": 1.5},
            )

        # Meta-optimizer
        meta_box = mpatches.FancyBboxPatch(
            (9.5, 2.8), 2.2, 2.4,
            boxstyle="round,pad=0.15", facecolor="#E3F2FD",
            edgecolor="#1565C0", linewidth=2,
        )
        ax.add_patch(meta_box)
        ax.text(10.6, 4.3, "Meta-Optimizer", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1565C0")
        ax.text(10.6, 3.5, "Allocates across\nsub-portfolios",
                ha="center", va="center", fontsize=8, color="#424242")

        ax.text(10.6, 7.3, "Stage 2: Meta-Optimization",
                ha="center", fontsize=11, fontweight="bold", color="#424242")

        # Arrow to final weights
        ax.annotate(
            "", xy=(12.5, 4.0), xytext=(11.7, 4.0),
            arrowprops={"arrowstyle": "-|>", "color": "#424242", "lw": 2},
        )

        # Final ensemble weights
        final_box = mpatches.FancyBboxPatch(
            (12.5, 3.0), 1.3, 2.0,
            boxstyle="round,pad=0.15", facecolor="#E8F5E9",
            edgecolor="#2E7D32", linewidth=2,
        )
        ax.add_patch(final_box)
        ax.text(13.15, 4.3, "Ensemble", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#2E7D32")
        ax.text(13.15, 3.7, r"$\mathbf{w}^*$", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#2E7D32")

        # Formula at bottom
        ax.text(
            7.0, 0.5,
            r"$\mathbf{w}^* = \sum_{k=1}^{K} \alpha_k \, \mathbf{w}_k$"
            r"   where $\boldsymbol{\alpha}$ minimizes portfolio risk",
            ha="center", fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#FFF9C4",
                "edgecolor": "#F9A825",
            },
        )

        fig.suptitle(
            "Ensemble Stacking: Diversifying Across Optimization Models",
            fontsize=13, fontweight="bold", y=0.98,
        )

        print("  Fig 83: Ensemble stacking schematic (pure patches)")

        plt.tight_layout()
        self._save(fig)
