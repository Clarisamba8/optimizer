"""Fig53OpinionPooling — density overlay of expert views and consensus."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from theory.figures._base import FigureGenerator


class Fig53OpinionPooling(FigureGenerator):
    """Density plot: 3 expert posteriors + base prior + consensus mixture.

    Expert 1: bullish (mu=12%, sigma=3%)
    Expert 2: bearish (mu=2%, sigma=4%)
    Expert 3: neutral (mu=7%, sigma=2.5%)
    Base prior: market equilibrium (mu=8%, sigma=5%)

    Consensus: linear pooling with pi = [0.35, 0.25, 0.20], residual 0.20
    to base prior.
    """

    @property
    def name(self) -> str:
        return "fig_53_opinion_pooling"

    def generate(self) -> None:
        x = np.linspace(-10, 25, 500)

        # Expert distributions (annualised % returns)
        experts = [
            ("Expert 1 (bullish)", 12.0, 3.0, "#2196F3", 0.35),
            ("Expert 2 (bearish)", 2.0, 4.0, "#FF5722", 0.25),
            ("Expert 3 (neutral)", 7.0, 2.5, "#4CAF50", 0.20),
        ]
        prior_mu, prior_sigma = 8.0, 5.0
        prior_weight = 1.0 - sum(e[4] for e in experts)  # 0.20

        fig, ax = plt.subplots(figsize=(10, 6))

        # Base prior (dashed)
        prior_pdf = stats.norm.pdf(x, prior_mu, prior_sigma)
        ax.plot(x, prior_pdf, color="#9E9E9E", ls="--", lw=2,
                label=f"Base prior (mu={prior_mu}%, pi={prior_weight:.2f})")

        # Expert posteriors
        consensus_pdf = prior_weight * prior_pdf
        for name, mu, sigma, color, pi in experts:
            pdf = stats.norm.pdf(x, mu, sigma)
            ax.fill_between(x, pdf, alpha=0.12, color=color)
            ax.plot(x, pdf, color=color, lw=1.5, ls=":",
                    label=f"{name}: mu={mu}%, sigma={sigma}%, pi={pi:.2f}")
            consensus_pdf += pi * pdf

        # Consensus (bold black)
        ax.plot(x, consensus_pdf, color="black", lw=3, alpha=0.9,
                label="Consensus (linear pooling)")
        ax.fill_between(x, consensus_pdf, alpha=0.08, color="black")

        # Consensus mean
        consensus_mean = (
            prior_weight * prior_mu
            + sum(e[4] * e[1] for e in experts)
        )
        ax.axvline(consensus_mean, color="black", ls=":", lw=1)
        ax.text(
            consensus_mean + 0.3, ax.get_ylim()[1] * 0.6,
            f"Consensus\nmean={consensus_mean:.1f}%",
            fontsize=9, color="black",
        )

        ax.set_xlabel("Annualised Return (%)")
        ax.set_ylabel("Density")
        ax.set_title(
            "Opinion Pooling: Consensus from Conflicting Expert Views\n"
            "Linear pooling with credibility weights",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(-10, 25)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        self._save(fig)
