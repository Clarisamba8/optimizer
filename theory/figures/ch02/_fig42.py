"""Fig42DMMvsHMM — qualitative comparison of DMM vs HMM regime detection.

Optional dependency
-------------------
This figure requires ``torch`` and ``pyro-ppl``, which are **NOT** listed
in ``pyproject.toml``.  Install manually if needed::

    pip install torch pyro-ppl

If the import fails, :meth:`Fig42DMMvsHMM.generate` prints a warning and
saves an informational placeholder PNG with installation instructions
instead of raising an exception.

Design rationale
----------------
Previous versions used a scatter plot comparing HMM log-likelihood/T
(exact) against DMM ELBO/T (variational lower bound).  This comparison
is **unfair by construction**: the ELBO is always below the true
log-marginal-likelihood by the KL divergence, so all points land below
the diagonal regardless of model quality.

This version instead shows a **qualitative** comparison on a single asset:
  - Panel A:  Returns with HMM discrete regime shading (hard switching).
  - Panel B:  Returns with DMM continuous latent state evolution (smooth).
  - Panel C:  Overlay of HMM bear probability vs DMM latent regime signal.

This highlights the key structural difference: HMM enforces discrete
regime switches while DMM allows gradual transitions through a
continuous latent space.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.moments._hmm import HMMConfig, fit_hmm
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_MAX_ASSETS_POOL = 50  # most-complete assets for NaN-safe data
_TRADING_DAYS = 252

try:
    from optimizer.moments._dmm import DMMConfig, fit_dmm as _fit_dmm

    _DMM_AVAILABLE = True
except ImportError:
    _DMM_AVAILABLE = False


class Fig42DMMvsHMM(FigureGenerator):
    """Qualitative DMM vs HMM regime detection on a single asset.

    Panel A:  Asset returns with HMM discrete regime shading
              (2-state Gaussian HMM, hard Viterbi-decoded switching).
    Panel B:  Same returns with DMM latent regime signal shown as a
              continuous color gradient (no discrete switches).
    Panel C:  Overlay of HMM smoothed bear-state probability (step-like)
              vs DMM latent regime signal (smooth) — highlights the
              structural difference between discrete and continuous
              latent state models.

    The asset is automatically selected as the one with the highest
    HMM regime separation (largest difference in regime means), ensuring
    the figure shows clear, pedagogically useful regime transitions.

    .. note::
        Requires ``pip install torch pyro-ppl`` — these are NOT in
        pyproject.toml.  DMM produces diagonal covariance only.
    """

    @property
    def name(self) -> str:
        return "fig_42_dmm_vs_hmm"

    def generate(self) -> None:
        if not _DMM_AVAILABLE:
            self._save_placeholder()
            return

        prices = self._prices

        # Select most-complete assets to maximise time series length.
        non_nan_counts = prices.notna().sum()
        top_assets = non_nan_counts.nlargest(_MAX_ASSETS_POOL).index
        p_subset = prices[top_assets]

        returns = clean_returns(
            prices_to_returns(p_subset.ffill()).dropna()
        ).dropna()

        # Use cross-section mean return as regime signal (same as Fig 39).
        # Individual stocks are too noisy — the HMM oscillates between
        # regimes almost daily.  The cross-asset mean diversifies away
        # idiosyncratic noise and produces clear, sustained regime periods.
        mean_returns = returns.mean(axis=1).to_frame("mean_return")
        T = len(mean_returns)
        n_assets = returns.shape[1]
        print(
            f"  Fig 42: DMM vs HMM on {n_assets}-asset cross-section mean "
            f"({T} days)"
        )

        hmm_cfg = HMMConfig(n_states=2, random_state=42)
        try:
            best_hmm = fit_hmm(mean_returns, hmm_cfg)
        except Exception as exc:
            warnings.warn(f"HMM failed: {exc}", stacklevel=2)
            self._save_placeholder()
            return

        asset_ret = mean_returns
        best_col = f"{n_assets}-asset mean"

        # ── Fit DMM ────────────────────────────────────────────────────────
        dmm_cfg = _dmm_config_univariate()
        try:
            dmm_result = _fit_dmm(asset_ret, dmm_cfg)
        except Exception as exc:
            warnings.warn(f"DMM failed for {best_col}: {exc}", stacklevel=2)
            self._save_placeholder()
            return

        # ── Extract signals ────────────────────────────────────────────────
        dates = asset_ret.index
        ret_vals = asset_ret.values.flatten() * 100  # daily return in %

        # HMM: sort states so state-0 = bear (lower mean)
        state_means = best_hmm.regime_means.values.flatten()
        ordered = list(np.argsort(state_means))
        bear_idx = ordered[0]
        hmm_bear_prob = best_hmm.smoothed_probs.values[:, bear_idx]

        # DMM: L2 norm of latent means as "regime intensity" signal.
        # Rescale to [0, 1] for comparison with HMM probability.
        latent_means = dmm_result.latent_means.values
        latent_norm = np.linalg.norm(latent_means, axis=1)
        # Invert if correlated with bull (high norm = calm) so that
        # high signal = bear for consistent comparison with HMM.
        corr = np.corrcoef(hmm_bear_prob, latent_norm)[0, 1]
        if corr < 0:
            latent_norm = latent_norm.max() - latent_norm
        # Min-max rescale to [0, 1]
        ln_min, ln_max = latent_norm.min(), latent_norm.max()
        if ln_max > ln_min:
            dmm_signal = (latent_norm - ln_min) / (ln_max - ln_min)
        else:
            dmm_signal = np.full_like(latent_norm, 0.5)

        # ── Plot ───────────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Panel A: Returns + HMM regime shading
        ax1 = axes[0]
        ax1.plot(dates, ret_vals, color="#212121", linewidth=0.6, alpha=0.8)
        dominant = best_hmm.smoothed_probs.values.argmax(axis=1)
        ordered_dominant = np.array(
            [ordered.index(d) for d in dominant], dtype=int
        )
        _shade_regimes(
            ax1, dates, ordered_dominant,
            colors=["#2196F3", "#F44336"],
        )
        ax1.axhline(0, color="#757575", linewidth=0.6, linestyle=":")
        ax1.set_ylabel("Daily return (%)")
        ax1.set_title(
            f"Panel A: {best_col} — HMM Discrete Regime Detection "
            "(hard switching)"
        )
        import matplotlib.patches as mpatches
        ax1.legend(
            handles=[
                mpatches.Patch(color="#2196F3", alpha=0.3, label="Bear / High-vol"),
                mpatches.Patch(color="#F44336", alpha=0.3, label="Bull / Low-vol"),
            ],
            loc="upper right", fontsize=9,
        )

        # Panel B: Returns + DMM continuous signal as background gradient
        ax2 = axes[1]
        ax2.plot(dates, ret_vals, color="#212121", linewidth=0.6, alpha=0.8)
        # Colour background by DMM signal intensity (bear-like = blue)
        _shade_continuous(ax2, dates, dmm_signal, cmap_name="coolwarm_r")
        ax2.axhline(0, color="#757575", linewidth=0.6, linestyle=":")
        ax2.set_ylabel("Daily return (%)")
        ax2.set_title(
            f"Panel B: {best_col} — DMM Continuous Latent Regime Signal "
            "(smooth transitions)"
        )

        # Panel C: Overlay of both signals
        ax3 = axes[2]
        ax3.plot(
            dates, hmm_bear_prob, color="#2196F3", linewidth=1.5,
            label="HMM P(bear)", alpha=0.9,
        )
        ax3.plot(
            dates, dmm_signal, color="#C62828", linewidth=1.2,
            linestyle="--", label="DMM regime signal (rescaled)", alpha=0.8,
        )
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_ylabel("Regime signal")
        ax3.set_xlabel("Date")
        ax3.set_title(
            "Panel C: HMM Bear Probability (discrete) vs DMM Latent Signal (continuous)"
        )
        ax3.legend(loc="upper right", fontsize=9)

        fig.suptitle(
            "HMM vs DMM Regime Detection: Discrete Switching vs "
            "Continuous Latent States\n"
            "NOTE: DMM requires pip install torch pyro-ppl "
            "(not in pyproject.toml); diagonal covariance only",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)

    def _save_placeholder(self) -> None:
        """Save an informational PNG when torch/pyro-ppl are not installed."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        msg = (
            "Figure 42: DMM vs HMM Regime Detection Comparison\n\n"
            "This figure requires optional dependencies:\n\n"
            "    pip install torch pyro-ppl\n\n"
            "These packages are NOT included in pyproject.toml because:\n"
            "  - torch is large (~2 GB) and GPU-platform-specific\n"
            "  - pyro-ppl requires torch as a base\n\n"
            "Once installed, re-run:\n"
            "    python -m theory.figures.ch02.generate_figures\n\n"
            "Note: DMM produces DIAGONAL covariance only.\n"
        )
        ax.text(
            0.5, 0.5, msg,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFF9C4", edgecolor="#F9A825"),
        )
        ax.set_title(
            "Fig 42 — DMM vs HMM (torch + pyro-ppl required)",
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dmm_config_univariate() -> object:
    """Return a DMMConfig sized for univariate financial return series.

    The default DMMConfig (z_dim=16, emission_dim=64, rnn_dim=128) has
    thousands of parameters — massively overparameterized for a single
    asset time series of ~1000 observations.  This compact config uses
    smaller hidden dimensions and more epochs for better convergence.

    Only called when DMM is available; isolated to avoid import errors
    at module level.
    """
    from optimizer.moments._dmm import DMMConfig  # noqa: PLC0415

    return DMMConfig(
        z_dim=4,
        emission_dim=16,
        transition_dim=16,
        rnn_dim=32,
        num_epochs=500,
        annealing_epochs=100,
        random_state=42,
    )


def _shade_regimes(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    dominant: np.ndarray,
    colors: list[str],
) -> None:
    """Fill axis background with discrete regime colour spans."""
    date_series = pd.Series(dates)
    n = len(dominant)
    start = 0
    for i in range(1, n + 1):
        if i == n or dominant[i] != dominant[start]:
            ax.axvspan(
                date_series.iloc[start],
                date_series.iloc[min(i, n - 1)],
                alpha=0.15,
                color=colors[dominant[start]],
                linewidth=0,
            )
            start = i


def _shade_continuous(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    signal: np.ndarray,
    cmap_name: str = "coolwarm_r",
) -> None:
    """Shade axis background with a continuous colour gradient.

    Each day gets a vertical strip coloured by the signal intensity.
    Uses a strided approach (groups of 5 days) for performance.
    """
    cmap = plt.get_cmap(cmap_name)
    stride = 5  # group days for performance
    date_series = pd.Series(dates)
    n = len(signal)
    for i in range(0, n - stride, stride):
        mean_val = float(np.mean(signal[i : i + stride]))
        ax.axvspan(
            date_series.iloc[i],
            date_series.iloc[min(i + stride, n - 1)],
            alpha=0.2,
            color=cmap(mean_val),
            linewidth=0,
        )
