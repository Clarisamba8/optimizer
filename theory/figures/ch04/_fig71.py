"""Fig71RegimeBudgetRotation — stacked area of regime-conditional risk budgets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.moments._hmm import HMMConfig, fit_hmm
from optimizer.optimization._regime_risk import compute_regime_budget
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_MAX_ASSETS_FOR_MEAN = 50
_SECTOR_GROUPS = [
    "Technology", "Healthcare", "Financials",
    "Consumer Cyclical", "Industrials", "Defensive",
]
_SECTOR_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800",
    "#E91E63", "#9C27B0", "#607D8B",
]


class Fig71RegimeBudgetRotation(FigureGenerator):
    """Stacked area: sector budget allocations over time driven by regime probs.

    Shows how budgets rotate from cyclical-heavy (expansion) to
    defensive-heavy (contraction) as regime probabilities shift.
    """

    @property
    def name(self) -> str:
        return "fig_71_regime_budget_rotation"

    def generate(self) -> None:
        prices = self._prices

        # Build univariate signal
        non_nan_counts = prices.notna().sum()
        top_assets = non_nan_counts.nlargest(_MAX_ASSETS_FOR_MEAN).index
        p_subset = prices[top_assets]

        returns = clean_returns(
            prices_to_returns(p_subset.ffill()).dropna()
        ).dropna()

        mean_returns = returns.mean(axis=1).to_frame("mean_return")

        config = HMMConfig(n_states=2, random_state=42)
        result = fit_hmm(mean_returns, config)

        dates = result.filtered_probs.index

        # Sort states: state-0 = bear (low mean), state-1 = bull (high mean)
        state_means = result.regime_means["mean_return"].values
        ordered_states = list(np.argsort(state_means))

        # Define regime-conditional budgets for 6 sector groups
        n_sectors = len(_SECTOR_GROUPS)

        # Expansion budget: overweight cyclicals
        b_expansion = np.array([0.25, 0.15, 0.20, 0.20, 0.15, 0.05])

        # Contraction budget: overweight defensives
        b_contraction = np.array([0.10, 0.20, 0.10, 0.05, 0.10, 0.45])

        regime_budgets = [b_contraction, b_expansion]  # bear=0, bull=1

        # Compute blended budgets over time
        budget_ts = np.zeros((len(dates), n_sectors))

        for t in range(len(dates)):
            gamma = result.filtered_probs.iloc[t].values[ordered_states]
            budget_ts[t] = compute_regime_budget(regime_budgets, gamma)

        print(
            f"  Fig 71: {len(dates)} days, {n_sectors} sectors, "
            f"budget range [{budget_ts.min():.2%}, {budget_ts.max():.2%}]"
        )

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.stackplot(
            dates,
            budget_ts.T,
            labels=_SECTOR_GROUPS,
            colors=_SECTOR_COLORS,
            alpha=0.8,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Risk Budget Allocation")
        ax.set_xlabel("Date")
        ax.set_title(
            "Regime-Conditional Risk Budget Rotation Across Sectors",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8, ncol=3)

        plt.tight_layout()
        self._save(fig)
