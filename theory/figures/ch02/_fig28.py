"""Fig28SampleMeanInstability — error bars of sample mean across window lengths."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import EmpiricalMu

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

# Number of assets to display (pick the ones with the most data)
_N_ASSETS = 7

# Window lengths in trading days and their display labels
_WINDOWS: list[tuple[int, str]] = [
    (60, "T=60"),
    (120, "T=120"),
    (252, "T=252"),
]

# Bar width and group offsets for grouped bar layout
_BAR_WIDTH = 0.22
_OFFSETS = [-_BAR_WIDTH, 0.0, _BAR_WIDTH]

# Colors per window
_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]


class Fig28SampleMeanInstability(FigureGenerator):
    """Grouped error-bar chart of sample mean instability across rolling windows.

    For each window length (T=60, T=120, T=252), takes the most recent T
    observations of clean arithmetic returns, fits :class:`EmpiricalMu`
    (annualised), and overlays ±1.96 × SE error bars to visualise how badly
    determined individual asset means are at short horizons.

    SE is approximated as :math:`\\hat{\\sigma} / \\sqrt{T}` where
    :math:`\\hat{\\sigma}` is the annualised cross-sectional standard deviation
    of the annualised mean estimator, computed from the per-asset daily vol and
    the standard error formula in return space.
    """

    @property
    def name(self) -> str:
        return "fig_28_sample_mean_instability"

    def generate(self) -> None:
        prices = self._prices

        # Use the full available history for cleaning, then slice per window
        returns_all = clean_returns(prices_to_returns(prices.ffill()).dropna()).dropna()

        # Filter to assets with annualised vol in [15%, 45%] — excludes micro-caps
        # and leveraged instruments that would dominate the chart with extreme bars
        annual_vol = returns_all.std() * np.sqrt(252)
        vol_mask = (annual_vol >= 0.15) & (annual_vol <= 0.45)
        filtered = returns_all.loc[:, vol_mask]

        # Fallback: relax upper bound if not enough assets pass the filter
        if filtered.shape[1] < _N_ASSETS:
            filtered = returns_all

        # From filtered set, pick the _N_ASSETS with the most complete history
        asset_counts = filtered.notna().sum().nlargest(_N_ASSETS)
        tickers = asset_counts.index.tolist()
        returns_all = returns_all[tickers]

        short_names = [
            t.replace("_US_EQ", "").replace("_EQ", "") for t in tickers
        ]

        print(
            f"  Fig 28: {len(tickers)} assets — windows "
            + ", ".join(str(w) for w, _ in _WINDOWS)
        )

        fig, ax = plt.subplots(figsize=(13, 5.5))
        x_positions = np.arange(len(tickers))

        for (t_days, label), offset, color in zip(  # noqa: B905
            _WINDOWS, _OFFSETS, _COLORS
        ):
            window_returns = returns_all.tail(t_days).dropna(axis=1, how="any")

            # Only keep tickers still present after the window dropna
            available = [t for t in tickers if t in window_returns.columns]
            if not available:
                continue

            wr = window_returns[available]

            # Fit EmpiricalMu — result is daily; annualise by x252
            emp_mu = EmpiricalMu()
            emp_mu.fit(wr)
            mu_daily = emp_mu.mu_  # shape (N,)

            # Per-asset daily std for SE calculation
            sigma_daily = wr.std().values  # shape (N,)

            # Annualise: mean x 252, SE = (sigma / sqrt(T)) x 252
            mu_ann = mu_daily * 252 * 100
            se_ann = (sigma_daily / np.sqrt(t_days)) * 252 * 100
            ci_half = 1.96 * se_ann

            # Map available tickers back to full x_positions
            x_idx = np.array(
                [i for i, t in enumerate(tickers) if t in available]
            )
            ax.bar(
                x_idx + offset,
                mu_ann,
                width=_BAR_WIDTH,
                color=color,
                alpha=0.75,
                label=label,
                zorder=3,
            )
            ax.errorbar(
                x_idx + offset,
                mu_ann,
                yerr=ci_half,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4,
                zorder=4,
            )

        ax.axhline(0.0, color="black", lw=0.8, ls=":")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(short_names, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Annualised Return (%)")
        ax.set_xlabel("Asset")
        ax.legend(title="Window", fontsize=9)
        ax.set_title(
            "Sample Mean Instability: 95% Confidence Intervals Across Window Lengths\n"
            f"({_N_ASSETS} assets (15–45% ann. vol filter), "
            "error bars = +/-1.96 x SE, SE = sigma/sqrt(T) x 252)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig)
