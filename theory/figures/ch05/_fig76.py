"""Fig76RobustEllipse — 2D uncertainty ellipses for different kappa values."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_RNG_SEED = 42


class Fig76RobustEllipse(FigureGenerator):
    """2D uncertainty ellipses around mu_hat for kappa = 0.5, 1.0, 2.0.

    Shows how larger kappa produces a wider uncertainty region, leading
    the robust optimizer to select more conservative weights.
    """

    @property
    def name(self) -> str:
        return "fig_76_robust_ellipse"

    def generate(self) -> None:
        mu_hat = np.array([0.10, 0.06])
        s_mu = np.array([[0.010, 0.002], [0.002, 0.008]])

        # Eigendecomposition for ellipse drawing
        eigvals, eigvecs = np.linalg.eigh(s_mu)

        kappas = [0.5, 1.0, 2.0]
        colors = ["#4CAF50", "#FF9800", "#E91E63"]
        labels = [
            r"$\kappa = 0.5$ (aggressive)",
            r"$\kappa = 1.0$ (moderate)",
            r"$\kappa = 2.0$ (conservative)",
        ]

        fig, ax = plt.subplots(figsize=(10, 8))

        theta = np.linspace(0, 2 * np.pi, 200)

        for kappa, color, label in zip(kappas, colors, labels, strict=True):
            # Ellipse: mu_hat + kappa * s_mu^{1/2} * [cos, sin]
            w = kappa * np.sqrt(eigvals[:, np.newaxis]) * np.array(
                [np.cos(theta), np.sin(theta)]
            )
            ellipse_pts = mu_hat[:, np.newaxis] + eigvecs @ w
            ax.plot(
                ellipse_pts[0], ellipse_pts[1],
                color=color, linewidth=2, label=label,
            )
            ax.fill(
                ellipse_pts[0], ellipse_pts[1],
                color=color, alpha=0.08,
            )

        # Mark mu_hat
        ax.plot(
            mu_hat[0], mu_hat[1], "k*", markersize=15, zorder=5,
            label=r"$\hat{\mu}$ (point estimate)",
        )

        # Worst-case mu for each kappa (direction of portfolio weight)
        # For a portfolio w = [0.6, 0.4], worst case is mu_hat - kappa * s_mu @ w / norm
        w_nominal = np.array([0.6, 0.4])
        direction = s_mu @ w_nominal
        direction = direction / np.linalg.norm(direction)

        for kappa, color in zip(kappas, colors, strict=True):
            worst_mu = mu_hat - kappa * np.sqrt(eigvals.max()) * direction * 0.5
            ax.plot(
                worst_mu[0], worst_mu[1], "x",
                color=color, markersize=10, markeredgewidth=2.5,
            )
            ax.annotate(
                "", xy=(worst_mu[0], worst_mu[1]),
                xytext=(mu_hat[0], mu_hat[1]),
                arrowprops={
                    "arrowstyle": "->", "color": color,
                    "lw": 1.5, "linestyle": "--",
                },
            )

        # Weight vector annotation
        scale = 0.08
        ax.annotate(
            "", xy=(mu_hat[0] + w_nominal[0] * scale,
                    mu_hat[1] + w_nominal[1] * scale),
            xytext=(mu_hat[0], mu_hat[1]),
            arrowprops={"arrowstyle": "-|>", "color": "#212121", "lw": 2},
        )
        ax.text(
            mu_hat[0] + w_nominal[0] * scale + 0.005,
            mu_hat[1] + w_nominal[1] * scale + 0.005,
            r"$\mathbf{w}$", fontsize=12, fontweight="bold",
        )

        ax.set_xlabel(r"$\mu_1$ (Asset 1 Expected Return)")
        ax.set_ylabel(r"$\mu_2$ (Asset 2 Expected Return)")
        ax.set_title(
            "Robust Optimization: Hedging Against the Worst-Case Expected Returns",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        print("  Fig 76: 2D robust uncertainty ellipses for kappa = 0.5, 1.0, 2.0")

        plt.tight_layout()
        self._save(fig)
