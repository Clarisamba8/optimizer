"""Fig27PreselectionFunnel — waterfall chart of the pre-selection pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa

from skfolio.pre_selection import DropCorrelated, DropZeroVariance
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns


class Fig27PreselectionFunnel(FigureGenerator):
    """Waterfall chart: institutional-grade 3yr portfolio pre-selection funnel.

    Five-stage funnel showing genuine universe reduction in a 3yr optimisation
    window using real DB data.  Each stage maps to a concrete sklearn/skfolio
    transformer in the production pre-selection pipeline:

    0. Full DB universe          — all active instruments
    1. 5yr complete history      — >=1200 genuine trading days (load_prices filter)
    2. Data quality + zero-var   — DataValidator + OutlierTreater + DropZeroVariance
    3. Decorrelation rho > 0.75  — DropCorrelated removes redundant pairs
    4. Top-200 by Sharpe         — final selection for optimisation

    Parameters
    ----------
    prices:
        Wide price DataFrame passed from the orchestrator.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        Connection string used only for Stage 0 raw-universe count query.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        output_dir: Path,
        db_url: str,
    ) -> None:
        super().__init__(prices, output_dir)
        self._db_url = db_url

    @property
    def name(self) -> str:
        return "fig_27_preselection_funnel"

    def generate(self) -> None:
        prices = self._prices

        # Stage 0: Full DB universe
        engine = sa.create_engine(self._db_url)
        n_raw = int(
            pd.read_sql(
                "SELECT COUNT(*) FROM instruments WHERE is_active = true", engine
            ).iloc[0, 0]
        )
        print(f"  Stage 0 - Full DB universe: {n_raw}")

        # Stage 1: 5yr complete history (enforced by PriceLoader min_days=1200)
        n1 = prices.shape[1]
        print(f"  Stage 1 - 5yr history (>=1200 genuine days): {n1}")

        # Slice to 3-year optimisation window (~756 trading days)
        THREE_YR = 756
        prices_3yr = prices.iloc[-THREE_YR:] if len(prices) > THREE_YR else prices
        print(
            f"  3yr window: {prices_3yr.index.min().date()} -> "
            f"{prices_3yr.index.max().date()} ({len(prices_3yr)} rows)"
        )

        # Stage 2: Data quality + zero-variance
        returns_q = clean_returns(prices_to_returns(prices_3yr.ffill()).dropna()).dropna()
        dz = DropZeroVariance()
        dz.fit(returns_q)
        cols2 = returns_q.columns[dz.get_support()]
        returns2 = pd.DataFrame(dz.transform(returns_q), index=returns_q.index, columns=cols2)
        n2 = returns2.shape[1]
        print(f"  Stage 2 - Data quality + zero-variance: {n2}")

        # Stage 3: Decorrelation rho > 0.75
        print(f"  Running DropCorrelated(0.75) on {n2} assets - may take ~60s ...")
        dc = DropCorrelated(threshold=0.75)
        dc.fit(returns2)
        cols3 = returns2.columns[dc.get_support()]
        returns3 = pd.DataFrame(dc.transform(returns2), index=returns2.index, columns=cols3)
        n3 = returns3.shape[1]
        print(f"  Stage 3 - Decorrelation rho>0.75: {n3}")

        # Stage 4: Top-200 by annualised Sharpe ratio
        k = 200
        n4 = min(k, n3)
        print(f"  Stage 4 - Top-{k} by annualised Sharpe: {n4}")
        print(f"  Full pipeline: {n_raw} -> {n1} -> {n2} -> {n3} -> {n4}")

        stages = [
            "Full DB\nUniverse",
            "5yr Complete\nHistory",
            "Data Quality\n& Zero-Variance",
            "Decorrelation\nrho > 0.75",
            f"Top-{k}\nby Sharpe",
        ]
        counts = [n_raw, n1, n2, n3, n4]
        colors_bar = ["#607D8B", "#2196F3", "#00BCD4", "#FF9800", "#4CAF50"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # ---- absolute counts ----
        bars = ax1.bar(range(len(stages)), counts, color=colors_bar,
                       edgecolor="white", lw=0.8, width=0.6)
        for bar, count in zip(bars, counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f"{count:,}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        for i in range(1, len(counts)):
            delta = counts[i] - counts[i - 1]
            if delta < 0:
                mid_y = (counts[i] + counts[i - 1]) / 2
                ax1.annotate(
                    f"-{abs(delta):,}",
                    xy=(i - 0.5, mid_y),
                    fontsize=8,
                    color="#C62828",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C62828", lw=0.8),
                )
            elif delta == 0:
                ax1.annotate(
                    "passed",
                    xy=(i - 0.5, counts[i] * 1.05),
                    fontsize=7,
                    color="#388E3C",
                    ha="center",
                    va="bottom",
                )
        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels(stages, fontsize=8.5)
        ax1.set_ylabel("Number of Assets")
        ax1.set_title("Absolute Asset Count per Stage")
        ax1.set_ylim(0, max(counts) * 1.18)

        # ---- percentage retention ----
        pct = [c / counts[0] * 100 for c in counts]
        bars2 = ax2.bar(range(len(stages)), pct, color=colors_bar,
                        edgecolor="white", lw=0.8, width=0.6)
        for bar, p in zip(bars2, pct):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{p:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels(stages, fontsize=8.5)
        ax2.set_ylabel("Percentage of Original Universe (%)")
        ax2.set_title("Universe Retention Rate per Stage")
        ax2.set_ylim(0, 120)
        ax2.axhline(100, color="grey", lw=0.8, ls=":")

        fig.suptitle(
            "Pre-Selection Pipeline: Institutional-Grade Universe Reduction\n"
            f"(3-Year Optimisation Window: "
            f"{prices_3yr.index.min().date()} -> {prices_3yr.index.max().date()})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
