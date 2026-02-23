"""Fig70RegimeRisk — dual-axis time series: returns + regime-driven risk weights."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.moments._hmm import HMMConfig, fit_hmm
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_MAX_ASSETS_FOR_MEAN = 50
_REGIME_COLORS = ["#2196F3", "#F44336"]


class Fig70RegimeRisk(FigureGenerator):
    """Dual-axis: returns with regime shading + risk measure weight area chart.

    Shows smooth transition from variance-dominant to CVaR-dominant
    risk measure during crisis periods, driven by HMM filtered probs.
    """

    @property
    def name(self) -> str:
        return "fig_70_regime_risk"

    def generate(self) -> None:
        prices = self._prices

        # Select most-complete assets
        non_nan_counts = prices.notna().sum()
        top_assets = non_nan_counts.nlargest(_MAX_ASSETS_FOR_MEAN).index
        p_subset = prices[top_assets]

        returns = clean_returns(
            prices_to_returns(p_subset.ffill()).dropna()
        ).dropna()

        # Univariate mean-return series as regime signal
        mean_returns = returns.mean(axis=1).to_frame("mean_return")

        config = HMMConfig(n_states=2, random_state=42)
        result = fit_hmm(mean_returns, config)

        dates = result.smoothed_probs.index

        # Sort states by mean return: state-0 = bear (stress)
        state_means = result.regime_means["mean_return"].values
        ordered_states = list(np.argsort(state_means))

        # Filtered probs for the ordered states
        probs = result.filtered_probs.values[:, ordered_states]
        bear_prob = probs[:, 0]  # stress state probability

        # Cumulative returns for top panel
        cum_ret = (1 + mean_returns["mean_return"]).cumprod()
        cum_ret_aligned = cum_ret.loc[dates]

        print(
            f"  Fig 70: {len(dates)} days, "
            f"bear regime = {(bear_prob > 0.5).mean():.1%} of time"
        )

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        # Panel A: Cumulative returns with regime shading
        ax1.plot(dates, cum_ret_aligned.values, color="#212121", linewidth=0.8)

        # Shade by dominant regime
        dominant = probs.argmax(axis=1)
        _shade_regimes(ax1, dates, dominant, _REGIME_COLORS)

        ax1.set_ylabel("Cumulative Return")
        ax1.set_title("Panel A: Market Returns with HMM Regime Shading", fontsize=10)
        patches = [
            mpatches.Patch(
                facecolor=_REGIME_COLORS[0], alpha=0.3, label="Bull / Low-vol",
            ),
            mpatches.Patch(
                facecolor=_REGIME_COLORS[1], alpha=0.3, label="Bear / High-vol",
            ),
        ]
        ax1.legend(handles=patches, loc="upper left", fontsize=9)

        # Panel B: Risk measure weights (stacked area)
        variance_weight = 1 - bear_prob  # calm -> variance

        ax2.fill_between(
            dates, 0, variance_weight, color="#2196F3", alpha=0.6,
            label="Variance weight",
        )
        ax2.fill_between(
            dates, variance_weight, 1, color="#F44336", alpha=0.6,
            label="CVaR weight",
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Risk Measure Weight")
        ax2.set_xlabel("Date")
        ax2.set_title(
            "Panel B: Effective Risk Measure Weights (Variance vs CVaR)",
            fontsize=10,
        )
        ax2.legend(loc="upper right", fontsize=9)

        fig.suptitle(
            "Regime-Driven Risk Measure Transition: Variance to CVaR During Crises",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)


def _shade_regimes(
    ax: plt.Axes,
    dates: object,
    dominant: np.ndarray,
    colors: list[str],
) -> None:
    """Fill axis background with regime colour spans."""
    date_series = pd.Series(dates)
    n = len(dominant)
    start = 0
    for i in range(1, n + 1):
        if i == n or dominant[i] != dominant[start]:
            ax.axvspan(
                date_series.iloc[start],
                date_series.iloc[min(i, n - 1)],
                alpha=0.15,
                color=colors[dominant[start]],
                linewidth=0,
            )
            start = i
