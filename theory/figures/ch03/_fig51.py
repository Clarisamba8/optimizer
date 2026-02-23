"""Fig51EPReweight — stem plot showing EP scenario reweighting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_N_SCENARIOS = 200
_RNG_SEED = 42


class Fig51EPReweight(FigureGenerator):
    """Stem plot: uniform prior probabilities vs EP-reweighted probabilities.

    Shows how Entropy Pooling adjusts scenario weights to satisfy a mean
    view while minimising KL divergence from the uniform prior.

    Uses a synthetic EP-like reweighting (analytical KL solution for a
    single mean constraint) to avoid needing the full optimizer.
    """

    @property
    def name(self) -> str:
        return "fig_51_ep_reweight"

    def generate(self) -> None:
        rng = np.random.default_rng(_RNG_SEED)

        # Generate synthetic scenarios
        scenarios = rng.normal(0.005, 0.03, _N_SCENARIOS)

        # Uniform prior
        p_prior = np.ones(_N_SCENARIOS) / _N_SCENARIOS

        # Analytical minimum-KL solution for a single mean equality constraint:
        # p*_s = p0_s * exp(lambda * x_s) / Z
        # We solve for lambda such that sum(p* * x) = target_mean
        target_mean = 0.01  # moderate bullish view

        # Binary search for the Lagrange multiplier
        lam_lo, lam_hi = -100.0, 100.0
        for _ in range(100):
            lam_mid = (lam_lo + lam_hi) / 2
            log_w = lam_mid * scenarios
            log_w -= log_w.max()  # numerical stability
            w = np.exp(log_w)
            p_star = w / w.sum()
            current_mean = np.sum(p_star * scenarios)
            if current_mean < target_mean:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

        # Final probabilities
        log_w = lam_mid * scenarios
        log_w -= log_w.max()
        w = np.exp(log_w)
        p_star = w / w.sum()

        prior_mean = np.mean(scenarios)
        posterior_mean = np.sum(p_star * scenarios)

        print(
            f"  Fig 51: prior mean = {prior_mean:.4f}, "
            f"posterior mean = {posterior_mean:.4f}, "
            f"target = {target_mean:.4f}"
        )

        # Sort by scenario value for cleaner visual
        sort_idx = np.argsort(scenarios)
        p_prior_s = p_prior[sort_idx]
        p_star_s = p_star[sort_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Panel 1: Prior (uniform)
        ax1.bar(
            range(_N_SCENARIOS), p_prior_s * 1000,
            color="#9E9E9E", alpha=0.6, width=1.0,
        )
        ax1.axhline(
            1000 / _N_SCENARIOS, color="#333", ls="--", lw=1,
            label=f"Uniform: 1/{_N_SCENARIOS} = {1/_N_SCENARIOS:.4f}",
        )
        ax1.set_ylabel("Probability (x1000)")
        ax1.set_title("Prior: Uniform Scenario Probabilities", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.set_ylim(0, max(p_prior_s * 1000) * 1.3)

        # Panel 2: Posterior (EP-reweighted)
        colors = np.where(
            p_star_s > 1 / _N_SCENARIOS * 1.5, "#2196F3",
            np.where(p_star_s < 1 / _N_SCENARIOS * 0.5, "#FF5722", "#9E9E9E")
        )
        ax2.bar(
            range(_N_SCENARIOS), p_star_s * 1000,
            color=colors, alpha=0.7, width=1.0,
        )
        ax2.axhline(
            1000 / _N_SCENARIOS, color="#333", ls="--", lw=1,
            label="Original uniform weight",
        )
        ax2.set_ylabel("Probability (x1000)")
        ax2.set_xlabel(
            "Scenarios (sorted by return, left=worst, right=best)"
        )
        ax2.set_title(
            f"Posterior: EP-Reweighted (mean view = {target_mean:.1%})",
            fontsize=10,
        )
        ax2.legend(fontsize=9)

        fig.suptitle(
            "Entropy Pooling Reweights Scenarios to Satisfy Views\n"
            f"Prior mean = {prior_mean:.2%} → Posterior mean = {posterior_mean:.2%} "
            f"(target = {target_mean:.1%})",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
