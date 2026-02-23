"""Fig91Pipeline — full pipeline block diagram."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from theory.figures._base import FigureGenerator


class Fig91Pipeline(FigureGenerator):
    """End-to-end pipeline architecture block diagram.

    Pure matplotlib.patches — no data needed.
    """

    @property
    def name(self) -> str:
        return "fig_91_pipeline"

    def generate(self) -> None:
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")

        # Pipeline stages
        stages = [
            ("Prices\n→ Returns", "#F5F5F5", "#616161", 0.3),
            ("SelectComplete", "#E3F2FD", "#1565C0", 2.5),
            ("DropZeroVar", "#E3F2FD", "#1565C0", 4.3),
            ("DropCorrelated", "#E3F2FD", "#1565C0", 6.1),
            ("SelectKExtremes", "#E3F2FD", "#1565C0", 7.9),
            ("Prior\n(μ + Σ)", "#FFF3E0", "#E65100", 9.9),
            ("Optimizer", "#E8F5E9", "#2E7D32", 11.9),
            ("Weights", "#FCE4EC", "#C62828", 14.0),
        ]

        box_width = 1.6
        box_height = 1.8
        y_center = 3.5

        for name, face_color, edge_color, x_pos in stages:
            box = mpatches.FancyBboxPatch(
                (x_pos, y_center - box_height / 2), box_width, box_height,
                boxstyle="round,pad=0.15", facecolor=face_color,
                edgecolor=edge_color, linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(x_pos + box_width / 2, y_center, name,
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=edge_color)

        # Arrows between stages
        arrow_props = {"arrowstyle": "-|>", "color": "#424242", "lw": 1.5}
        x_positions = [s[3] for s in stages]
        for i in range(len(x_positions) - 1):
            ax.annotate(
                "", xy=(x_positions[i + 1], y_center),
                xytext=(x_positions[i] + box_width, y_center),
                arrowprops=arrow_props,
            )

        # Bracket for pre-selection pipeline
        ax.annotate(
            "", xy=(2.5, y_center + box_height / 2 + 0.5),
            xytext=(9.5, y_center + box_height / 2 + 0.5),
            arrowprops={"arrowstyle": "-", "color": "#1565C0", "lw": 1.5},
        )
        ax.text(6.0, y_center + box_height / 2 + 0.8,
                "Pre-Selection Pipeline (sklearn Pipeline)",
                ha="center", fontsize=9, fontstyle="italic", color="#1565C0")

        # Parameter paths
        param_examples = [
            (3.3, 1.0, '"drop_correlated__threshold"'),
            (6.9, 1.0, '"select_k_extremes__k"'),
            (10.7, 1.0, '"optimizer__l2_coef"'),
            (10.7, 0.5, '"prior_estimator__mu_estimator__alpha"'),
        ]
        for x, y, text in param_examples:
            ax.text(x, y, text, ha="center", fontsize=7, fontfamily="monospace",
                    color="#757575")

        ax.text(7.0, 0.2, "sklearn get_params() parameter paths",
                ha="center", fontsize=8, fontstyle="italic", color="#9E9E9E")

        # Factor returns metadata (y route)
        ax.annotate(
            "Factor Returns\n(metadata routing)",
            xy=(10.7, y_center - box_height / 2),
            xytext=(10.7, y_center - box_height / 2 - 1.2),
            fontsize=8, ha="center", color="#E65100",
            arrowprops={
                "arrowstyle": "-|>", "color": "#E65100",
                "lw": 1.2, "linestyle": "--",
            },
        )

        fig.suptitle(
            "End-to-End Pipeline Architecture",
            fontsize=13, fontweight="bold", y=0.98,
        )

        print("  Fig 91: pipeline block diagram (pure patches)")

        plt.tight_layout()
        self._save(fig)
