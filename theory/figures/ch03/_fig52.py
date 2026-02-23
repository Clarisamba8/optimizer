"""Fig52EPViewTypes — multi-panel distributions for different EP view types."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from theory.figures._base import FigureGenerator

_N_SCENARIOS = 5000
_RNG_SEED = 42


class Fig52EPViewTypes(FigureGenerator):
    """Multi-panel (2x3) density plots showing EP effect per view type.

    Each panel shows the prior (empirical) distribution vs the posterior
    after applying a single view type:
    (a) Prior (empirical)
    (b) After mean view (shifted)
    (c) After variance view (wider)
    (d) After skewness view (asymmetric)
    (e) After CVaR view (heavier left tail)

    Uses analytical minimum-KL tilts on synthetic scenarios rather than
    running the full EP optimizer.
    """

    @property
    def name(self) -> str:
        return "fig_52_ep_view_types"

    def generate(self) -> None:
        rng = np.random.default_rng(_RNG_SEED)

        # Base scenarios: slightly right-skewed returns
        scenarios = rng.normal(0.005, 0.03, _N_SCENARIOS)
        p_uniform = np.ones(_N_SCENARIOS) / _N_SCENARIOS

        panels = [
            ("(a) Prior\n(empirical)", p_uniform),
            ("(b) Mean view\n(E[r]=2%)", self._tilt_mean(scenarios, 0.02)),
            ("(c) Variance view\n(Vol x2)", self._tilt_variance(scenarios, 0.06)),
            ("(d) Skewness view\n(skew=-1.5)", self._tilt_skew(scenarios, -1.5)),
            ("(e) CVaR view\n(CVaR₅%=8%)", self._tilt_cvar(scenarios, 0.08)),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes_flat = axes.flatten()

        x_grid = np.linspace(-0.12, 0.12, 200)

        for idx, (title, weights) in enumerate(panels):
            ax = axes_flat[idx]

            # Weighted histogram density (robust with extreme weights)
            density = self._weighted_density(scenarios, weights, x_grid)

            # Prior density for comparison
            density_prior = self._weighted_density(
                scenarios, p_uniform, x_grid,
            )

            ax.fill_between(
                x_grid * 100, density_prior, alpha=0.2, color="#9E9E9E",
                label="Prior",
            )
            ax.plot(x_grid * 100, density_prior, color="#9E9E9E",
                    ls="--", lw=1)

            _colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
            color = _colors[idx]
            ax.fill_between(
                x_grid * 100, density, alpha=0.35, color=color,
                label="Posterior" if idx > 0 else "Distribution",
            )
            ax.plot(x_grid * 100, density, color=color, lw=1.5)

            # Stats
            w_mean = np.average(scenarios, weights=weights) * 100
            w_std = np.sqrt(np.average(
                (scenarios - np.average(scenarios, weights=weights))**2,
                weights=weights,
            )) * 100
            ax.text(
                0.97, 0.95,
                f"Mean={w_mean:.2f}%\nStd={w_std:.2f}%",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            )

            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Return (%)" if idx >= 2 else "")
            ax.set_ylabel("Density" if idx % 3 == 0 else "")
            ax.legend(fontsize=7, loc="upper left")

        # Hide the 6th subplot
        axes_flat[5].set_visible(False)

        fig.suptitle(
            "Entropy Pooling: How Different View Types Reshape the Return Distribution",
            fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        self._save(fig)

    @staticmethod
    def _weighted_density(
        scenarios: np.ndarray,
        weights: np.ndarray,
        x_grid: np.ndarray,
        n_bins: int = 80,
    ) -> np.ndarray:
        """Compute a smooth weighted density via histogram + Gaussian filter."""
        from scipy.ndimage import gaussian_filter1d

        hist, bin_edges = np.histogram(
            scenarios, bins=n_bins, weights=weights, density=True,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist.astype(float), sigma=2.0)
        return np.interp(x_grid, bin_centers, smoothed)

    @staticmethod
    def _tilt_mean(
        scenarios: np.ndarray, target_mean: float,
    ) -> np.ndarray:
        """KL-optimal tilt for a mean constraint."""
        lam_lo, lam_hi = -200.0, 200.0
        for _ in range(100):
            lam = (lam_lo + lam_hi) / 2
            log_w = lam * scenarios
            log_w -= log_w.max()
            w = np.exp(log_w)
            p = w / w.sum()
            if np.sum(p * scenarios) < target_mean:
                lam_lo = lam
            else:
                lam_hi = lam
        log_w = lam * scenarios
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    @staticmethod
    def _tilt_variance(
        scenarios: np.ndarray, target_std: float,
    ) -> np.ndarray:
        """KL-optimal tilt for a variance constraint (exponential quadratic)."""
        mu = np.mean(scenarios)
        target_var = target_std**2
        lam_lo, lam_hi = -5000.0, 5000.0
        for _ in range(100):
            lam = (lam_lo + lam_hi) / 2
            log_w = lam * (scenarios - mu)**2
            log_w -= log_w.max()
            w = np.exp(log_w)
            p = w / w.sum()
            var_current = np.sum(p * (scenarios - np.sum(p * scenarios))**2)
            if var_current < target_var:
                lam_lo = lam
            else:
                lam_hi = lam
        log_w = lam * (scenarios - mu)**2
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    @staticmethod
    def _tilt_skew(
        scenarios: np.ndarray, target_skew: float,
    ) -> np.ndarray:
        """KL-optimal tilt for a skewness constraint."""
        mu = np.mean(scenarios)
        std = np.std(scenarios)
        z = (scenarios - mu) / std
        lam_lo, lam_hi = -50.0, 50.0
        for _ in range(100):
            lam = (lam_lo + lam_hi) / 2
            log_w = lam * z**3
            log_w -= log_w.max()
            w = np.exp(log_w)
            p = w / w.sum()
            p_mu = np.sum(p * scenarios)
            p_std = np.sqrt(np.sum(p * (scenarios - p_mu)**2))
            skew = np.sum(p * ((scenarios - p_mu) / max(p_std, 1e-10))**3)
            if skew > target_skew:
                lam_hi = lam
            else:
                lam_lo = lam
        log_w = lam * z**3
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    @staticmethod
    def _tilt_cvar(
        scenarios: np.ndarray, target_cvar: float, alpha: float = 0.05,
    ) -> np.ndarray:
        """Tilt to approximate a CVaR view by upweighting left tail scenarios."""
        # Exponential tilt on negative returns to increase tail weight
        lam_lo, lam_hi = -500.0, 0.0
        for _ in range(100):
            lam = (lam_lo + lam_hi) / 2
            log_w = lam * scenarios
            log_w -= log_w.max()
            w = np.exp(log_w)
            p = w / w.sum()
            # Compute weighted CVaR
            sort_idx = np.argsort(scenarios)
            cum_p = np.cumsum(p[sort_idx])
            tail_idx = sort_idx[cum_p <= alpha]
            if len(tail_idx) == 0:
                tail_idx = sort_idx[:1]
            tail_sum = max(np.sum(p[tail_idx]), 1e-10)
            cvar = -np.sum(
                p[tail_idx] * scenarios[tail_idx]
            ) / tail_sum
            if cvar < target_cvar:
                lam_hi = lam
            else:
                lam_lo = lam
        log_w = lam * scenarios
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()
