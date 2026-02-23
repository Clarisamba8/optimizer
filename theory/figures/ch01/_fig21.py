"""Fig21OutlierGroups — histogram with colour-coded outlier zones."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.preprocessing._outliers import OutlierTreater
from theory.figures._base import FigureGenerator


class Fig21OutlierGroups(FigureGenerator):
    """Histogram with colour-coded outlier zones using real + augmented data.

    Uses ANF (Abercrombie & Fitch) real returns as the base because its
    earnings-driven volatility naturally populates the Winsorize zone.
    The Remove zone (|z| >= 10) represents data-feed errors (price unit
    mismatches, corrupt entries) which don't appear in clean institutional
    data — so 5 synthetic errors at +-12-15 sigma are injected to make the
    zone visible.  The injection is explicit in the title and axis annotations.
    """

    @property
    def name(self) -> str:
        return "fig_21_outlier_groups"

    def generate(self) -> None:
        prices = self._prices

        preferred = "ANF_US_EQ"
        ticker = preferred if preferred in prices.columns else (
            prices.notna().sum().sort_values(ascending=False).index[0]
        )
        short = ticker.replace("_US_EQ", "").replace("p_EQ", "")

        raw = prices[[ticker]].dropna()
        returns = raw.pct_change().dropna()
        raw_vals = returns[ticker].values
        print(f"  Fig 21: using {ticker} ({len(raw_vals)} days of returns)")

        # Fit on real data to get mu and sigma for zone boundaries
        treater_real = OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)
        treater_real.fit(returns)
        mu = treater_real.mu_[ticker]
        sigma = treater_real.sigma_[ticker]

        # Inject 5 synthetic data errors at +-12-15 sigma to populate the Remove zone.
        # These represent real-world data problems: unit mismatches (GBX vs GBP),
        # corporate-action feed errors, or stale-price spikes.
        rng = np.random.default_rng(42)
        n_errors = 5
        err_positions = rng.choice(len(raw_vals), size=n_errors, replace=False)
        err_signs = np.array([+1, -1, +1, -1, +1])
        err_magnitudes = rng.uniform(12, 15, size=n_errors) * sigma * err_signs
        augmented_vals = raw_vals.copy()
        augmented_vals[err_positions] = mu + err_magnitudes

        # Re-fit and transform on the augmented series
        aug_df = pd.DataFrame({ticker: augmented_vals}, index=returns.index)
        treater = OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)
        treater.fit(aug_df)
        df_treated = treater.transform(aug_df)

        win_lo = mu - 3 * sigma
        win_hi = mu + 3 * sigma
        rem_lo = mu - 10 * sigma
        rem_hi = mu + 10 * sigma

        # Display range: clip to +-12 sigma so all three zone boundaries are visible
        x_lo = mu - 12 * sigma
        x_hi = mu + 12 * sigma

        n_winsorized = int(
            ((augmented_vals > win_hi) & (augmented_vals <= rem_hi)).sum()
            + ((augmented_vals < win_lo) & (augmented_vals >= rem_lo)).sum()
        )
        n_removed = int((np.abs(augmented_vals - mu) / sigma >= 10).sum())

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        n_bins = 70

        # ----- Before treatment -----
        ax = axes[0]
        clipped = np.clip(augmented_vals, x_lo, x_hi)
        ax.hist(clipped, bins=n_bins, range=(x_lo, x_hi),
                color="#9E9E9E", edgecolor="white", lw=0.3)

        ax.axvspan(x_lo,   rem_lo, alpha=0.18, color="#F44336")
        ax.axvspan(rem_hi, x_hi,   alpha=0.18, color="#F44336")
        ax.axvspan(rem_lo, win_lo, alpha=0.18, color="#FF9800")
        ax.axvspan(win_hi, rem_hi, alpha=0.18, color="#FF9800")
        ax.axvspan(win_lo, win_hi, alpha=0.12, color="#4CAF50")
        ax.axvline(win_lo, color="#4CAF50", lw=1.5, ls="--",
                   label=f"+-3sigma  (+-{3*sigma*100:.1f}%)")
        ax.axvline(win_hi, color="#4CAF50", lw=1.5, ls="--")
        ax.axvline(rem_lo, color="#F44336", lw=1.5, ls=":",
                   label=f"+-10sigma (+-{10*sigma*100:.1f}%)")
        ax.axvline(rem_hi, color="#F44336", lw=1.5, ls=":")

        n_keep = int((np.abs(augmented_vals - mu) / sigma < 3).sum())
        green_patch  = mpatches.Patch(color="#4CAF50", alpha=0.5,
                                      label=f"Keep   |z|<3  ({n_keep} obs)")
        orange_patch = mpatches.Patch(color="#FF9800", alpha=0.5,
                                      label=f"Winsor 3<=|z|<10 ({n_winsorized} obs)")
        red_patch    = mpatches.Patch(color="#F44336", alpha=0.5,
                                      label=f"Remove |z|>=10  ({n_removed} obs *)")
        ax.legend(handles=[green_patch, orange_patch, red_patch],
                  fontsize=8, loc="upper right")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.set_xlim(x_lo, x_hi)
        ax.set_title(
            f"Before Treatment  ({short}, {len(augmented_vals):,} obs)\n"
            f"* {n_errors} synthetic data errors injected at +-12-15 sigma"
        )

        # ----- After treatment -----
        ax2 = axes[1]
        treated_vals = df_treated[ticker].dropna().values
        ax2.hist(treated_vals, bins=n_bins, color="#2196F3", edgecolor="white", lw=0.3)
        ax2.axvspan(win_lo, win_hi, alpha=0.12, color="#4CAF50")
        ax2.axvline(win_lo, color="#4CAF50", lw=1.5, ls="--", label="+-3 sigma boundary")
        ax2.axvline(win_hi, color="#4CAF50", lw=1.5, ls="--")
        ax2.set_xlabel("Daily Return")
        ax2.set_ylabel("Frequency")
        ax2.set_title(
            f"After Treatment\n"
            f"{n_winsorized} winsorised -> clipped to +-3 sigma  |  "
            f"{n_removed} data errors -> NaN"
        )
        ax2.legend(fontsize=9)

        # Annotate winsorize spikes at the +-3 sigma boundary
        ymax2 = ax2.get_ylim()[1]
        ax2.annotate(
            f"winsorised\nspike ({n_winsorized})",
            xy=(win_hi, ymax2 * 0.6),
            fontsize=8,
            color="#E65100",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#E65100"),
            xytext=(win_hi + sigma, ymax2 * 0.6),
        )

        fig.suptitle(
            f"Three-Group Outlier Treatment: Remove / Winsorize / Keep  ({short})\n"
            f"Real returns (NYSE) + {n_errors} injected data-feed errors "
            f"to illustrate the Remove zone",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
