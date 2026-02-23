"""Fig12RegimeTiltHeatmap — Regime-conditional factor weight multipliers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.factors._config import (
    FactorGroupType,
    MacroRegime,
    RegimeTiltConfig,
)
from optimizer.factors._regime import get_regime_tilts
from theory.figures._base import FigureGenerator


class Fig12RegimeTiltHeatmap(FigureGenerator):
    """4x9 heatmap of regime tilt multipliers by regime and factor group.

    Pure configuration visualization — no DB or price data needed.

    Parameters
    ----------
    prices:
        Wide price DataFrame (unused but required by base class).
    output_dir:
        Directory where the generated PNG is saved.
    """

    @property
    def name(self) -> str:
        return "fig_12_regime_tilt_heatmap"

    def generate(self) -> None:
        config = RegimeTiltConfig()
        regimes = list(MacroRegime)
        groups = list(FactorGroupType)

        # Build the tilt matrix
        data = np.ones((len(regimes), len(groups)))
        for i, regime in enumerate(regimes):
            tilts = get_regime_tilts(regime, config)
            for j, group in enumerate(groups):
                data[i, j] = tilts.get(group, 1.0)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        vmax = max(abs(data.max() - 1.0), abs(data.min() - 1.0))
        im = ax.imshow(
            data,
            cmap="RdYlGn",
            aspect="auto",
            vmin=1.0 - vmax,
            vmax=1.0 + vmax,
        )

        # Annotate cells
        for i in range(len(regimes)):
            for j in range(len(groups)):
                val = data[i, j]
                color = "white" if abs(val - 1.0) > vmax * 0.6 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=9, color=color,
                    fontweight="bold",
                )

        # Labels
        group_labels = [g.value.replace("_", " ").title() for g in groups]
        regime_labels = [r.value.capitalize() for r in regimes]
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(group_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(regimes)))
        ax.set_yticklabels(regime_labels, fontsize=10)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Tilt Multiplier", fontsize=9)

        ax.set_title(
            "Regime-Conditional Factor Weight Multipliers\n"
            "(green > 1 = overweight, red < 1 = underweight)",
        )
        plt.tight_layout()
        self._save(fig)
