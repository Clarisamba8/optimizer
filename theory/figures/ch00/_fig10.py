"""Fig10NetAlpha — Net alpha sensitivity to transaction costs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_GROSS_ALPHA = 0.03  # 3% annualized gross alpha
_COST_BPS_VALUES = (10, 25, 50)  # round-trip cost scenarios
_TURNOVER_RANGE = np.linspace(0, 4.0, 200)  # 0% to 400% annual turnover
_COLORS = ("#4CAF50", "#FF9800", "#E91E63")


class Fig10NetAlpha(FigureGenerator):
    """Net alpha vs transaction cost for different cost-per-unit-turnover levels.

    Pure analytical figure — no database or price data required.  Shows how
    net alpha degrades as turnover increases for 10, 25, and 50 bps round-trip
    cost assumptions, with breakeven points annotated.

    Parameters
    ----------
    prices:
        Wide price DataFrame (unused, required by base class).
    output_dir:
        Directory where the generated PNG is saved.
    """

    @property
    def name(self) -> str:
        return "fig_10_net_alpha_vs_cost"

    def generate(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for cost_bps, color in zip(_COST_BPS_VALUES, _COLORS, strict=False):
            cost_fraction = cost_bps / 10_000
            net_alpha = _GROSS_ALPHA - _TURNOVER_RANGE * cost_fraction
            label = f"{cost_bps} bps round-trip"
            ax.plot(
                _TURNOVER_RANGE * 100,
                net_alpha * 100,
                color=color,
                lw=2,
                label=label,
            )

            # Breakeven: gross_alpha = turnover * cost → turnover = gross_alpha / cost
            breakeven_turnover = _GROSS_ALPHA / cost_fraction
            if breakeven_turnover <= _TURNOVER_RANGE[-1]:
                ax.plot(
                    breakeven_turnover * 100,
                    0,
                    "o",
                    color=color,
                    ms=7,
                    zorder=5,
                )
                ax.annotate(
                    f"{breakeven_turnover * 100:.0f}%",
                    xy=(breakeven_turnover * 100, 0),
                    xytext=(breakeven_turnover * 100 + 15, -0.4),
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "lw": 1.2,
                    },
                )

        # Gross alpha reference
        ax.axhline(
            _GROSS_ALPHA * 100,
            color="#9E9E9E",
            ls="--",
            lw=1.5,
            label=f"Gross alpha ({_GROSS_ALPHA * 100:.0f}%)",
        )
        ax.axhline(0, color="black", lw=0.6, ls=":")

        ax.set_xlabel("Annual One-Way Turnover (%)")
        ax.set_ylabel("Net Alpha (%)")
        ax.set_title(
            "Net Alpha Sensitivity to Transaction Costs\n"
            f"(Gross alpha = {_GROSS_ALPHA * 100:.0f}%, "
            "breakeven points marked)",
        )
        ax.set_xlim(0, _TURNOVER_RANGE[-1] * 100)
        ax.set_ylim(-3.5, 4.0)
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig)
