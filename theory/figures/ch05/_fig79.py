"""Fig79VineCopula — D-vine tree diagram for 5 assets."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from theory.figures._base import FigureGenerator


class Fig79VineCopula(FigureGenerator):
    """D-vine tree structure for 5 assets showing copula decomposition.

    Pure matplotlib.patches diagram — no data needed.
    """

    @property
    def name(self) -> str:
        return "fig_79_vine_copula"

    def generate(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 9)
        ax.set_aspect("equal")
        ax.axis("off")

        assets = ["SPY", "TLT", "GLD", "VGK", "EEM"]
        node_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
        copula_families = [
            ["Gaussian", "Student-t", "Clayton", "Gumbel"],
            ["Gaussian", "Student-t", "Clayton"],
            ["Gaussian", "Student-t"],
            ["Gaussian"],
        ]

        # Tree 1: all 5 nodes in a line, 4 edges
        tree1_y = 7.5
        tree1_xs = [1.5, 4.0, 6.5, 9.0, 11.5]

        ax.text(0.3, tree1_y, "Tree 1", fontsize=10, fontweight="bold",
                color="#424242", va="center")

        for i, (x, asset) in enumerate(zip(tree1_xs, assets, strict=True)):
            circle = mpatches.Circle(
                (x, tree1_y), 0.4,
                facecolor=node_colors[i], edgecolor="#424242",
                linewidth=1.5, alpha=0.8,
            )
            ax.add_patch(circle)
            ax.text(x, tree1_y, asset, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

        for i in range(4):
            mid_x = (tree1_xs[i] + tree1_xs[i + 1]) / 2
            ax.plot(
                [tree1_xs[i] + 0.4, tree1_xs[i + 1] - 0.4],
                [tree1_y, tree1_y],
                color="#424242", linewidth=2,
            )
            ax.text(
                mid_x, tree1_y + 0.35, copula_families[0][i],
                ha="center", fontsize=7, fontstyle="italic", color="#616161",
            )

        # Tree 2: 4 nodes (midpoints of tree 1 edges)
        tree2_y = 5.5
        tree2_xs = [2.75, 5.25, 7.75, 10.25]
        tree2_labels = [
            f"{assets[0]},{assets[1]}",
            f"{assets[1]},{assets[2]}",
            f"{assets[2]},{assets[3]}",
            f"{assets[3]},{assets[4]}",
        ]

        ax.text(0.3, tree2_y, "Tree 2", fontsize=10, fontweight="bold",
                color="#424242", va="center")

        for _i, (x, label) in enumerate(zip(tree2_xs, tree2_labels, strict=True)):
            box = mpatches.FancyBboxPatch(
                (x - 0.7, tree2_y - 0.3), 1.4, 0.6,
                boxstyle="round,pad=0.1", facecolor="#E3F2FD",
                edgecolor="#1565C0", linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(x, tree2_y, label, ha="center", va="center", fontsize=8)

        for i in range(3):
            mid_x = (tree2_xs[i] + tree2_xs[i + 1]) / 2
            ax.plot(
                [tree2_xs[i] + 0.7, tree2_xs[i + 1] - 0.7],
                [tree2_y, tree2_y],
                color="#1565C0", linewidth=1.5, linestyle="--",
            )
            ax.text(
                mid_x, tree2_y + 0.35, copula_families[1][i],
                ha="center", fontsize=7, fontstyle="italic", color="#1565C0",
            )

        # Tree 3: 3 nodes
        tree3_y = 3.5
        tree3_xs = [4.0, 6.5, 9.0]
        tree3_labels = [
            f"{assets[0]},{assets[2]}|{assets[1]}",
            f"{assets[1]},{assets[3]}|{assets[2]}",
            f"{assets[2]},{assets[4]}|{assets[3]}",
        ]

        ax.text(0.3, tree3_y, "Tree 3", fontsize=10, fontweight="bold",
                color="#424242", va="center")

        for _i, (x, label) in enumerate(zip(tree3_xs, tree3_labels, strict=True)):
            box = mpatches.FancyBboxPatch(
                (x - 1.0, tree3_y - 0.3), 2.0, 0.6,
                boxstyle="round,pad=0.1", facecolor="#FFF3E0",
                edgecolor="#E65100", linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(x, tree3_y, label, ha="center", va="center", fontsize=7.5)

        for i in range(2):
            mid_x = (tree3_xs[i] + tree3_xs[i + 1]) / 2
            ax.plot(
                [tree3_xs[i] + 1.0, tree3_xs[i + 1] - 1.0],
                [tree3_y, tree3_y],
                color="#E65100", linewidth=1.5, linestyle=":",
            )
            ax.text(
                mid_x, tree3_y + 0.35, copula_families[2][i],
                ha="center", fontsize=7, fontstyle="italic", color="#E65100",
            )

        # Tree 4: 2 nodes
        tree4_y = 1.5
        tree4_xs = [5.25, 7.75]
        tree4_labels = [
            f"{assets[0]},{assets[3]}|{assets[1]},{assets[2]}",
            f"{assets[1]},{assets[4]}|{assets[2]},{assets[3]}",
        ]

        ax.text(0.3, tree4_y, "Tree 4", fontsize=10, fontweight="bold",
                color="#424242", va="center")

        for _i, (x, label) in enumerate(zip(tree4_xs, tree4_labels, strict=True)):
            box = mpatches.FancyBboxPatch(
                (x - 1.3, tree4_y - 0.3), 2.6, 0.6,
                boxstyle="round,pad=0.1", facecolor="#FCE4EC",
                edgecolor="#C62828", linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(x, tree4_y, label, ha="center", va="center", fontsize=7)

        mid_x = (tree4_xs[0] + tree4_xs[1]) / 2
        ax.plot(
            [tree4_xs[0] + 1.3, tree4_xs[1] - 1.3],
            [tree4_y, tree4_y],
            color="#C62828", linewidth=1.5, linestyle="-.",
        )
        ax.text(
            mid_x, tree4_y + 0.35, copula_families[3][0],
            ha="center", fontsize=7, fontstyle="italic", color="#C62828",
        )

        # Summary annotation
        ax.text(
            7.0, 0.3,
            r"D-vine: $\binom{5}{2} = 10$ bivariate copulas across 4 trees",
            ha="center", fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#FFF9C4",
                "edgecolor": "#F9A825",
            },
        )

        fig.suptitle(
            "Vine Copula Decomposition: Tree Structure for 5 Assets",
            fontsize=13, fontweight="bold", y=0.98,
        )

        print("  Fig 79: D-vine tree diagram for 5 assets (pure patches)")

        plt.tight_layout()
        self._save(fig)
