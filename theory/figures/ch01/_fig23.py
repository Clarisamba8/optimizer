"""Fig23DataQuality — 3-panel data quality dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.preprocessing._validation import DataValidator
from theory.figures._base import FigureGenerator
from theory.figures._helpers import prices_to_returns


class Fig23DataQuality(FigureGenerator):
    """3-panel data quality dashboard using the full real universe."""

    @property
    def name(self) -> str:
        return "fig_23_data_quality"

    def generate(self) -> None:
        prices = self._prices
        print(f"  Fig 23: universe {prices.shape[1]} assets x {prices.shape[0]} days")

        # Run DataValidator on arithmetic returns
        returns = prices_to_returns(prices)
        validator = DataValidator(max_abs_return=10.0)
        validator.fit(returns)
        returns_clean = validator.transform(returns)

        missing_pct_before = returns.isna().mean(axis=0) * 100
        missing_pct_after = returns_clean.isna().mean(axis=0) * 100
        extreme_counts = (returns.abs() > 1.0).sum(axis=0)  # >100% daily move

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # (a) Missing data histogram
        ax = axes[0]
        bins = np.linspace(0, max(missing_pct_before.max(), 1), 40)
        ax.hist(missing_pct_before, bins=bins, color="#2196F3", edgecolor="white",
                lw=0.5, alpha=0.8, label="Before validation")
        ax.hist(missing_pct_after, bins=bins, color="#FF5722", edgecolor="white",
                lw=0.5, alpha=0.7, label="After validation (infs->NaN)")
        ax.set_xlabel("Missing Data (%)")
        ax.set_ylabel("Number of Assets")
        ax.set_title("(a) Missing Data Distribution\nby Asset")
        ax.legend(fontsize=9)

        # (b) Extreme return counts — top 15 assets
        ax2 = axes[1]
        top_assets = extreme_counts.nlargest(15)
        short_labels = [t.replace("p_EQ", "") for t in top_assets.index]
        ax2.bar(range(len(top_assets)), top_assets.values,
                color="#FF5722", edgecolor="white", lw=0.5)
        ax2.set_xticks(range(len(top_assets)))
        ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Days with |return| > 100%")
        ax2.set_title("(b) Extreme Return Frequency\n(top 15 affected assets)")

        # (c) Panel completeness by calendar quarter
        returns_clean.index = pd.to_datetime(returns_clean.index)
        quarterly_completeness = (
            returns_clean.resample("QE")
            .apply(lambda g: g.notna().to_numpy().mean() * 100)
            .mean(axis=1)
        )
        labels_q = [str(d.date())[:7] for d in quarterly_completeness.index]
        complete_vals = quarterly_completeness.values.tolist()
        missing_vals = [100 - c for c in complete_vals]
        ax3 = axes[2]
        ax3.bar(range(len(labels_q)), complete_vals, color="#4CAF50", label="Complete")
        ax3.bar(range(len(labels_q)), missing_vals, bottom=complete_vals,
                color="#FF5722", alpha=0.7, label="Missing")
        ax3.set_xticks(range(len(labels_q)))
        ax3.set_xticklabels(labels_q, rotation=45, ha="right", fontsize=7)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("Panel Completeness (%)")
        ax3.set_title("(c) Data Completeness by Quarter")
        ax3.legend(fontsize=9)

        fig.suptitle(
            f"Data Quality Assessment Dashboard\n"
            f"({prices.shape[1]}-asset multi-exchange universe, "
            f"{prices.index.min().date()} -> {prices.index.max().date()})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
