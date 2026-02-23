"""Fig46BLPosterior — grouped bar chart showing BL posterior at high/low confidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import EquilibriumMu, LedoitWolf
from skfolio.prior import BlackLitterman, EmpiricalPrior

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 756
_MAX_ASSETS = 10
_RISK_AVERSION = 2.5


class Fig46BLPosterior(FigureGenerator):
    """Grouped bar chart: equilibrium vs BL posterior at high/low confidence.

    Shows three series:
    (a) Equilibrium returns Pi (grey)
    (b) BL posterior with high-confidence views (blue)
    (c) BL posterior with low-confidence views (orange)
    """

    @property
    def name(self) -> str:
        return "fig_46_bl_posterior"

    def generate(self) -> None:
        prices = self._prices
        p_window = (
            prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        )
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        # Select assets
        last_prices = p_window[returns.columns].iloc[-1].sort_values(ascending=False)
        tickers = last_prices.index[:_MAX_ASSETS].tolist()
        returns = returns[tickers]
        n_assets = len(tickers)
        short_names = [
            t.replace("_US_EQ", "").replace("_EQ", "") for t in tickers
        ]

        print(f"  Fig 46: {n_assets} assets x {len(returns)} days")

        # Fit equilibrium prior to get Pi
        eq_prior = EmpiricalPrior(
            mu_estimator=EquilibriumMu(risk_aversion=_RISK_AVERSION),
            covariance_estimator=LedoitWolf(),
        )
        eq_prior.fit(returns)
        pi = eq_prior.return_distribution_.mu * 252 * 100  # annualised %

        # Define two views on the first two assets (daily returns for skfolio)
        v1_ticker = tickers[0]
        v2_ticker = tickers[1]
        view_ann_1 = 15.0   # 15% annualised
        view_ann_2 = -5.0   # -5% annualised
        views = [
            f"{v1_ticker} == {view_ann_1 / 100 / 252}",
            f"{v2_ticker} == {view_ann_2 / 100 / 252}",
        ]

        # High confidence BL (small tau → strong prior, but high view_confidences)
        bl_high = BlackLitterman(
            views=views,
            prior_estimator=EmpiricalPrior(
                mu_estimator=EquilibriumMu(risk_aversion=_RISK_AVERSION),
                covariance_estimator=LedoitWolf(),
            ),
            tau=0.05,
            view_confidences=[0.95, 0.95],
        )
        bl_high.fit(returns)
        mu_high = bl_high.return_distribution_.mu * 252 * 100

        # Low confidence BL
        bl_low = BlackLitterman(
            views=views,
            prior_estimator=EmpiricalPrior(
                mu_estimator=EquilibriumMu(risk_aversion=_RISK_AVERSION),
                covariance_estimator=LedoitWolf(),
            ),
            tau=0.05,
            view_confidences=[0.10, 0.10],
        )
        bl_low.fit(returns)
        mu_low = bl_low.return_distribution_.mu * 252 * 100

        # Plot
        x = np.arange(n_assets)
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(12, n_assets * 0.8), 5.5))

        ax.bar(x - width, pi, width, color="#9E9E9E", alpha=0.85,
               label=r"Equilibrium $\Pi$")
        ax.bar(x, mu_high, width, color="#2196F3", alpha=0.85,
               label="BL Posterior (high confidence)")
        ax.bar(x + width, mu_low, width, color="#FF9800", alpha=0.85,
               label="BL Posterior (low confidence)")

        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Annualised Return (%)")
        ax.set_title(
            "Black-Litterman Posterior: View Confidence Modulates the Tilt\n"
            f"Views: {short_names[0]}=15%, {short_names[1]}=-5% | "
            f"High conf=0.95, Low conf=0.10",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)

        # Annotate the view targets
        for idx, (_name, target) in enumerate(
            [(short_names[0], 15.0), (short_names[1], -5.0)]
        ):
            ax.annotate(
                f"View: {target:+.0f}%",
                xy=(idx, mu_high[idx]),
                xytext=(idx + 0.5, mu_high[idx] + 3),
                arrowprops={"arrowstyle": "->", "color": "#333", "lw": 0.8},
                fontsize=8, color="#333",
            )

        plt.tight_layout()
        self._save(fig)
