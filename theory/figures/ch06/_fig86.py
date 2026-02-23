"""Fig86CPCVFoldMatrix — CPCV fold assignment matrix for C(6,2)=15 combinations."""

from __future__ import annotations

from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_N_FOLDS = 6
_N_TEST = 2


class Fig86CPCVFoldMatrix(FigureGenerator):
    """15x6 matrix where each row is a CPCV combination, columns are folds.

    Color-coded: blue = train, green = test.
    """

    @property
    def name(self) -> str:
        return "fig_86_cpcv_fold_matrix"

    def generate(self) -> None:
        combos = list(combinations(range(_N_FOLDS), _N_TEST))
        n_combos = len(combos)

        # Build matrix: 0 = train, 1 = test
        matrix = np.zeros((n_combos, _N_FOLDS), dtype=int)
        for i, test_folds in enumerate(combos):
            for f in test_folds:
                matrix[i, f] = 1

        fig, ax = plt.subplots(figsize=(10, 9))

        train_color = (0.129, 0.588, 0.953, 0.7)   # #2196F3
        test_color = (0.298, 0.686, 0.314, 0.7)    # #4CAF50
        purge_color = (1.0, 0.596, 0.0, 0.5)       # #FF9800

        for i in range(n_combos):
            for j in range(_N_FOLDS):
                color = test_color if matrix[i, j] == 1 else train_color

                rect = mpatches.Rectangle(
                    (j, n_combos - i - 1), 1, 1,
                    facecolor=color, edgecolor="white", linewidth=1.5,
                )
                ax.add_patch(rect)

                # Add purge zones (cells adjacent to test cells)
                if matrix[i, j] == 0:
                    # Check if adjacent to a test fold
                    is_purge = False
                    if j > 0 and matrix[i, j - 1] == 1:
                        is_purge = True
                    if j < _N_FOLDS - 1 and matrix[i, j + 1] == 1:
                        is_purge = True

                    if is_purge:
                        # Add small purge indicator
                        purge_rect = mpatches.Rectangle(
                            (j, n_combos - i - 1), 1, 0.15,
                            facecolor=purge_color, edgecolor="none",
                        )
                        ax.add_patch(purge_rect)

        ax.set_xlim(0, _N_FOLDS)
        ax.set_ylim(0, n_combos)
        ax.set_xticks(np.arange(_N_FOLDS) + 0.5)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(_N_FOLDS)], fontsize=10)
        ax.set_yticks(np.arange(n_combos) + 0.5)
        ax.set_yticklabels(
            [f"Combo {n_combos - i}" for i in range(n_combos)], fontsize=8,
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=train_color, label="Train"),
            Patch(facecolor=test_color, label="Test"),
            Patch(facecolor=purge_color, label="Purge Zone"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  bbox_to_anchor=(1.15, 1.0), fontsize=9)

        ax.set_title(
            f"CPCV Fold Structure: All {n_combos} Train-Test Combinations "
            f"for {_N_FOLDS} Folds",
            fontsize=12, fontweight="bold",
        )

        print(
            f"  Fig 86: CPCV matrix {n_combos} combinations x {_N_FOLDS} folds"
        )

        plt.tight_layout()
        self._save(fig)
