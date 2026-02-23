"""Fig84WalkForward — walk-forward Gantt chart with rolling vs expanding."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from theory.figures._base import FigureGenerator


class Fig84WalkForward(FigureGenerator):
    """Timeline showing 5 walk-forward steps: rolling and expanding variants.

    Pure matplotlib — synthetic time ranges.
    """

    @property
    def name(self) -> str:
        return "fig_84_walk_forward"

    def generate(self) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        t_train = 252
        t_test = 63
        n_steps = 5

        train_color = "#2196F3"
        test_color = "#4CAF50"

        # --- Panel A: Rolling windows ---
        ax1.set_title("Rolling Window (Fixed Training Size)", fontsize=11,
                       fontweight="bold")
        for k in range(n_steps):
            train_start = k * t_test
            train_end = train_start + t_train
            test_start = train_end
            y = n_steps - k - 1

            # Train block
            ax1.barh(y, t_train, left=train_start, height=0.6,
                     color=train_color, alpha=0.7, edgecolor=train_color)
            # Test block
            ax1.barh(y, t_test, left=test_start, height=0.6,
                     color=test_color, alpha=0.7, edgecolor=test_color)
            # Labels
            ax1.text(train_start + t_train / 2, y, f"Train {k+1}",
                     ha="center", va="center", fontsize=8, color="white",
                     fontweight="bold")
            ax1.text(test_start + t_test / 2, y, f"Test {k+1}",
                     ha="center", va="center", fontsize=8, color="white",
                     fontweight="bold")

        ax1.set_yticks(range(n_steps))
        ax1.set_yticklabels([f"Step {n_steps - i}" for i in range(n_steps)],
                             fontsize=9)
        ax1.set_xlim(0, n_steps * t_test + t_train + 20)

        # --- Panel B: Expanding windows ---
        ax2.set_title("Expanding Window (Growing Training Size)", fontsize=11,
                       fontweight="bold")
        for k in range(n_steps):
            train_start = 0
            train_end = t_train + k * t_test
            test_start = train_end
            y = n_steps - k - 1

            ax2.barh(y, train_end - train_start, left=train_start, height=0.6,
                     color=train_color, alpha=0.7, edgecolor=train_color)
            ax2.barh(y, t_test, left=test_start, height=0.6,
                     color=test_color, alpha=0.7, edgecolor=test_color)
            ax2.text((train_start + train_end) / 2, y,
                     f"Train {k+1}\n({train_end} days)",
                     ha="center", va="center", fontsize=7, color="white",
                     fontweight="bold")
            ax2.text(test_start + t_test / 2, y, f"Test {k+1}",
                     ha="center", va="center", fontsize=8, color="white",
                     fontweight="bold")

        ax2.set_yticks(range(n_steps))
        ax2.set_yticklabels([f"Step {n_steps - i}" for i in range(n_steps)],
                             fontsize=9)
        ax2.set_xlabel("Trading Days")

        # Legend
        train_patch = mpatches.Patch(color=train_color, alpha=0.7,
                                      label=f"Training (T_train = {t_train})")
        test_patch = mpatches.Patch(color=test_color, alpha=0.7,
                                     label=f"Testing (T_test = {t_test})")
        ax1.legend(handles=[train_patch, test_patch], loc="upper right",
                   fontsize=9)

        fig.suptitle(
            "Walk-Forward Backtesting: Rolling vs Expanding Window Timeline",
            fontsize=13, fontweight="bold", y=0.98,
        )

        print(f"  Fig 84: walk-forward Gantt, {n_steps} steps")

        plt.tight_layout()
        self._save(fig)
