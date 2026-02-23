"""Fig69NCOSchematic — pure matplotlib.patches diagram of NCO two-stage flow."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from theory.figures._base import FigureGenerator


class Fig69NCOSchematic(FigureGenerator):
    """Pure patches diagram showing two-stage NCO decomposition.

    No data needed — just a schematic of how the universe is split into
    clusters, inner optimization runs per cluster, and outer optimization
    combines cluster portfolios.
    """

    @property
    def name(self) -> str:
        return "fig_69_nco_schematic"

    def generate(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")

        # Colors
        cluster_colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]

        # Stage 0: Universe box (left)
        universe_box = mpatches.FancyBboxPatch(
            (0.3, 1.5), 2.4, 4.0,
            boxstyle="round,pad=0.15", facecolor="#F5F5F5",
            edgecolor="#616161", linewidth=1.5,
        )
        ax.add_patch(universe_box)
        ax.text(1.5, 5.2, "Universe\n(100 assets)", ha="center", va="center",
                fontsize=10, fontweight="bold")

        # Draw small dots representing assets
        import numpy as np
        rng = np.random.default_rng(42)
        for i in range(40):
            x = rng.uniform(0.5, 2.5)
            y = rng.uniform(1.8, 5.1)
            cidx = i % 5
            ax.plot(x, y, "o", color=cluster_colors[cidx], markersize=3, alpha=0.6)

        # Arrow: Universe -> Clustering
        ax.annotate(
            "", xy=(3.5, 3.5), xytext=(2.8, 3.5),
            arrowprops={"arrowstyle": "-|>", "color": "#424242", "lw": 2},
        )
        ax.text(3.15, 3.9, "Cluster", ha="center", fontsize=8, fontstyle="italic")

        # Stage 1: Clusters (middle)
        cluster_names = ["Cluster 1\n(20 assets)", "Cluster 2\n(25 assets)",
                         "Cluster 3\n(18 assets)", "Cluster 4\n(22 assets)",
                         "Cluster 5\n(15 assets)"]
        cluster_y_positions = [5.8, 4.6, 3.4, 2.2, 1.0]

        pairs = zip(cluster_names, cluster_y_positions, strict=True)
        for i, (name, y_pos) in enumerate(pairs):
            box = mpatches.FancyBboxPatch(
                (3.8, y_pos - 0.4), 2.2, 0.8,
                boxstyle="round,pad=0.1", facecolor=cluster_colors[i],
                edgecolor=cluster_colors[i], alpha=0.25, linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(4.9, y_pos, name, ha="center", va="center", fontsize=8)

        # Stage 1 label
        ax.text(4.9, 6.7, "Stage 1: Inner Optimization",
                ha="center", fontsize=10, fontweight="bold", color="#424242")

        # Arrows: Clusters -> Composite assets
        for i, y_pos in enumerate(cluster_y_positions):
            ax.annotate(
                "", xy=(6.8, y_pos), xytext=(6.1, y_pos),
                arrowprops={"arrowstyle": "-|>", "color": cluster_colors[i], "lw": 1.5},
            )

        # Composite assets (small circles)
        for i, y_pos in enumerate(cluster_y_positions):
            circle = plt.Circle(
                (7.2, y_pos), 0.25,
                facecolor=cluster_colors[i], edgecolor="#424242",
                alpha=0.5, linewidth=1,
            )
            ax.add_patch(circle)
            ax.text(7.2, y_pos, f"C{i+1}", ha="center", va="center",
                    fontsize=8, fontweight="bold")

        ax.text(
            7.2, 6.7, "Composite\nAssets", ha="center", fontsize=9, fontstyle="italic",
        )

        # Arrow: Composites -> Outer optimization
        ax.annotate(
            "", xy=(8.5, 3.5), xytext=(7.5, 3.5),
            arrowprops={"arrowstyle": "-|>", "color": "#424242", "lw": 2},
        )

        # Stage 2: Outer optimization box
        outer_box = mpatches.FancyBboxPatch(
            (8.7, 2.0), 2.2, 3.0,
            boxstyle="round,pad=0.15", facecolor="#E3F2FD",
            edgecolor="#1565C0", linewidth=1.5,
        )
        ax.add_patch(outer_box)
        ax.text(9.8, 6.7, "Stage 2: Outer Optimization",
                ha="center", fontsize=10, fontweight="bold", color="#424242")
        ax.text(9.8, 4.2, "Outer\nOptimization", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1565C0")
        ax.text(9.8, 3.0, "Min Variance\nover 5 clusters\n(5x5 matrix)",
                ha="center", va="center", fontsize=8, color="#424242")

        # Arrow: Outer -> Final weights
        ax.annotate(
            "", xy=(11.8, 3.5), xytext=(11.0, 3.5),
            arrowprops={"arrowstyle": "-|>", "color": "#424242", "lw": 2},
        )

        # Final weights box
        final_box = mpatches.FancyBboxPatch(
            (12.0, 1.5), 1.7, 4.0,
            boxstyle="round,pad=0.15", facecolor="#E8F5E9",
            edgecolor="#2E7D32", linewidth=1.5,
        )
        ax.add_patch(final_box)
        ax.text(12.85, 5.2, "Final Weights", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#2E7D32")

        # Example weights
        example_weights = [
            ("C1: 30%", cluster_colors[0]),
            ("C2: 25%", cluster_colors[1]),
            ("C3: 20%", cluster_colors[2]),
            ("C4: 15%", cluster_colors[3]),
            ("C5: 10%", cluster_colors[4]),
        ]
        for i, (text, color) in enumerate(example_weights):
            y = 4.5 - i * 0.6
            ax.text(12.85, y, text, ha="center", va="center", fontsize=9, color=color,
                    fontweight="bold")

        # Formula at bottom
        ax.text(
            7.0, 0.2,
            r"$w_i^{\mathrm{final}} = w_i^{\mathrm{inner}} \times w_k^{\mathrm{outer}}$"
            "   (asset $i$ in cluster $k$)",
            ha="center", va="center", fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#FFF9C4",
                "edgecolor": "#F9A825",
            },
        )

        fig.suptitle(
            "NCO Two-Stage Decomposition: Inner and Outer Optimization",
            fontsize=13, fontweight="bold", y=0.98,
        )

        print("  Fig 69: NCO schematic diagram (pure patches)")

        plt.tight_layout()
        self._save(fig)
