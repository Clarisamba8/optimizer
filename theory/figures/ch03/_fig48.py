"""Fig48ConfidenceSensitivity — posterior return vs view confidence sweep."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import EquilibriumMu, LedoitWolf
from skfolio.prior import BlackLitterman, EmpiricalPrior

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 756
_ALPHAS = np.linspace(0.01, 1.0, 50)


class Fig48ConfidenceSensitivity(FigureGenerator):
    """Line chart: posterior return of a single asset vs view confidence alpha.

    For a single absolute view ("Asset X returns 15%"), sweeps alpha from 0.01
    to 1.0 and plots the smooth transition from equilibrium return to full
    view adoption.
    """

    @property
    def name(self) -> str:
        return "fig_48_confidence_sensitivity"

    def generate(self) -> None:
        prices = self._prices
        p_window = (
            prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        )
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        # Pick the first asset with enough data
        tickers = returns.columns.tolist()
        target_ticker = tickers[0]
        target_idx = 0
        short_name = target_ticker.replace("_US_EQ", "").replace("_EQ", "")

        print(f"  Fig 48: sweeping confidence for {short_name}")

        # Get equilibrium return
        eq_prior = EmpiricalPrior(
            mu_estimator=EquilibriumMu(risk_aversion=2.5),
            covariance_estimator=LedoitWolf(),
        )
        eq_prior.fit(returns)
        pi_target = eq_prior.return_distribution_.mu[target_idx] * 252 * 100

        view_target = 15.0  # 15% annualised
        view_str = f"{target_ticker} == {view_target / 100 / 252}"

        posterior_returns = []
        for alpha in _ALPHAS:
            bl = BlackLitterman(
                views=[view_str],
                prior_estimator=EmpiricalPrior(
                    mu_estimator=EquilibriumMu(risk_aversion=2.5),
                    covariance_estimator=LedoitWolf(),
                ),
                tau=0.05,
                view_confidences=[float(alpha)],
            )
            bl.fit(returns)
            mu_post = bl.return_distribution_.mu[target_idx] * 252 * 100
            posterior_returns.append(mu_post)

        posterior_returns = np.array(posterior_returns)

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ax.plot(_ALPHAS, posterior_returns, color="#2196F3", lw=2.5,
                label=f"Posterior E[r] for {short_name}")
        ax.axhline(pi_target, color="#9E9E9E", ls="--", lw=1.5,
                    label=rf"Equilibrium $\Pi$ = {pi_target:.1f}%")
        ax.axhline(view_target, color="#FF5722", ls=":", lw=1.5,
                    label=f"View target = {view_target:.0f}%")

        # Fill between equilibrium and posterior
        ax.fill_between(
            _ALPHAS, pi_target, posterior_returns,
            alpha=0.15, color="#2196F3",
        )

        ax.set_xlabel(r"View confidence $\alpha_k$ (Idzorek)")
        ax.set_ylabel("Annualised Posterior Return (%)")
        ax.set_title(
            rf"View Confidence Sensitivity: How $\alpha_k$ Controls Posterior Tilt"
            f"\nAbsolute view: {short_name} = {view_target:.0f}% | "
            rf"$\tau$ = 0.05",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig)
