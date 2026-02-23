"""Fig39HMMRegimes — three-panel HMM regime detection figure."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.moments._hmm import HMMConfig, fit_hmm
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_MAX_ASSETS_FOR_MEAN = 50  # most-complete assets for cross-section mean
_REGIME_COLORS = ["#2196F3", "#F44336", "#FF9800", "#9C27B0"]


class Fig39HMMRegimes(FigureGenerator):
    """Three-panel HMM regime detection figure.

    Panel 1:  Cross-asset mean return series with regime background shading.
    Panel 2:  Smoothed posterior state probabilities as a stacked area chart.
    Panel 3:  Transition matrix heatmap with annotated probability values.

    A 2-state Gaussian HMM is fitted on a univariate mean-return series
    derived from the cross-asset average of the cleaned return panel.
    Uses all available data (no truncation) with the most-complete assets
    to maximise the time span and capture multiple market regimes.
    """

    @property
    def name(self) -> str:
        return "fig_39_hmm_regimes"

    def generate(self) -> None:
        prices = self._prices

        # Select most-complete assets to minimise NaN-induced data loss.
        # Using all 2255 assets with .dropna() loses ~60% of the time series.
        non_nan_counts = prices.notna().sum()
        top_assets = non_nan_counts.nlargest(_MAX_ASSETS_FOR_MEAN).index
        p_subset = prices[top_assets]

        returns = clean_returns(
            prices_to_returns(p_subset.ffill()).dropna()
        ).dropna()

        # Univariate mean-return series as regime signal
        mean_returns = returns.mean(axis=1).to_frame("mean_return")
        print(
            f"  Fig 39: HMM on {len(mean_returns)} days "
            f"({returns.shape[1]} assets cross-section mean)"
        )

        config = HMMConfig(n_states=2, random_state=42)
        result = fit_hmm(mean_returns, config)

        dates = result.smoothed_probs.index
        n_states = result.smoothed_probs.shape[1]
        colors = _REGIME_COLORS[:n_states]

        # Sort states by mean return so state-0 is "bearish"
        state_means = result.regime_means["mean_return"].values
        ordered_states = list(np.argsort(state_means))  # ascending: bear first
        state_labels = ["Bear / High-vol", "Bull / Low-vol"][:n_states]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

        # ── Panel 1: mean return series + regime shading ─────────────────────
        ax1 = axes[0]
        series = mean_returns["mean_return"]
        ax1.plot(dates, series.values * 100, color="#212121", linewidth=0.7, alpha=0.9)

        # Shade background by dominant smoothed state
        dominant = result.smoothed_probs.values.argmax(axis=1)
        # Map to ordered index
        ordered_dominant = np.array(
            [ordered_states.index(d) for d in dominant], dtype=int
        )
        _shade_regimes(ax1, dates, ordered_dominant, colors)

        ax1.axhline(0, color="#757575", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Daily return (%)")
        ax1.set_title("Panel A: Mean Portfolio Return with HMM Regime Shading")
        _add_legend_patches(ax1, colors, state_labels, n_states)

        # ── Panel 2: smoothed posterior probabilities (stacked area) ──────────
        ax2 = axes[1]
        probs_ordered = result.smoothed_probs.values[:, ordered_states]
        ax2.stackplot(
            dates,
            probs_ordered.T,
            labels=state_labels,
            colors=colors,
            alpha=0.85,
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(state | data)")
        ax2.set_title("Panel B: Smoothed Posterior State Probabilities")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_xlabel("Date")

        # ── Panel 3: transition matrix heatmap ───────────────────────────────
        ax3 = axes[2]
        trans = result.transition_matrix[np.ix_(ordered_states, ordered_states)]
        im = ax3.imshow(trans, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        for i in range(n_states):
            for j in range(n_states):
                ax3.text(
                    j, i, f"{trans[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if trans[i, j] > 0.5 else "#212121",
                )

        ax3.set_xticks(range(n_states))
        ax3.set_yticks(range(n_states))
        ax3.set_xticklabels([f"→ {s}" for s in state_labels])
        ax3.set_yticklabels([f"{s} →" for s in state_labels])
        ax3.set_title("Panel C: HMM Transition Matrix  A[i→j]")
        # No shared x-axis for panel 3 (it's a matrix, not a time series)
        ax3.set_xlabel("")

        fig.suptitle(
            "HMM Regime Detection\n"
            f"(2-state Gaussian HMM, {len(dates)}-day window, "
            f"{returns.shape[1]} assets)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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


def _add_legend_patches(
    ax: plt.Axes,
    colors: list[str],
    labels: list[str],
    n_states: int,
) -> None:
    """Add coloured patch legend for regime shading."""
    patches = [
        mpatches.Patch(facecolor=colors[i], alpha=0.4, label=labels[i])
        for i in range(n_states)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
