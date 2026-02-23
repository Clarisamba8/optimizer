"""Fig40HMMBlended — rolling HMM-blended vs empirical moments comparison.

Critical design note
--------------------
This module uses :class:`~optimizer.moments._hmm.HMMBlendedCovariance`
(the skfolio-compatible estimator) rather than the standalone
:func:`~optimizer.moments._hmm.blend_moments_by_regime` helper.

The two differ in an important way:

    ``blend_moments_by_regime`` computes only the within-regime weighted sum:

        Σ = Σ_s p_s · Σ_s

    ``HMMBlendedCovariance`` applies the *full law of total variance*:

        Σ = Σ_s p_s · [Σ_s + (μ_s - μ)(μ_s - μ)ᵀ]

The second term ``(μ_s - μ)(μ_s - μ)ᵀ`` is the cross-state mean-dispersion
contribution.  When regime means differ materially (e.g. bull vs bear), this
term inflates the blended covariance substantially, producing appropriately
conservative risk estimates during periods of regime uncertainty.

**Always use** ``HMMBlendedCovariance`` for optimizer inputs.  Only use
``blend_moments_by_regime`` for lightweight diagnostic inspection.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skfolio.moments import EmpiricalCovariance, EmpiricalMu

from optimizer.moments._hmm import HMMBlendedCovariance, HMMBlendedMu, HMMConfig
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 504   # 2-year rolling window (gives ~15 steps with typical data)
_STEP = 21           # monthly step
_LOOKBACK = 252 * 5  # 5-year lookback for rolling loop
_TRADING_DAYS = 252
_MAX_ASSETS = 8      # cap for HMM stability (fewer assets → less regime-mean noise)


class Fig40HMMBlended(FigureGenerator):
    """Two-panel rolling comparison: HMM-blended vs empirical moments.

    Panel 1:  Rolling annualised expected return (cross-asset mean)
              — HMM-blended (solid blue) vs Empirical (dashed grey).
    Panel 2:  Rolling annualised volatility (sqrt of mean diagonal)
              — HMM-blended covariance (solid red) vs Empirical (dashed grey).

    Both are computed on a 2-year rolling window stepped monthly over the
    last 5 years of available data.

    .. warning::
        Uses ``HMMBlendedCovariance`` (full law of total variance), NOT
        ``blend_moments_by_regime``.  The class includes the cross-state
        mean-dispersion term; the function does not.  Results differ — always
        use the class for optimizer inputs.
    """

    @property
    def name(self) -> str:
        return "fig_40_hmm_blended"

    def generate(self) -> None:  # noqa: PLR0912
        prices = self._prices

        # Select most-complete assets first to minimise NaN-induced data loss.
        # Using all 2255 assets with .dropna() loses ~60% of time series.
        _N_POOL = 50
        non_nan_counts = prices.notna().sum()
        top_assets = non_nan_counts.nlargest(_N_POOL).index
        p_pool = prices[top_assets]

        lookback = min(_LOOKBACK, len(p_pool))
        p_window = p_pool.iloc[-lookback:]
        returns_full = clean_returns(
            prices_to_returns(p_window.ffill()).dropna()
        ).dropna()

        # Select lowest-volatility assets for stable HMM fits.
        ann_vol = returns_full.std() * np.sqrt(_TRADING_DAYS)
        stable_cols = ann_vol.nsmallest(_MAX_ASSETS).index
        returns_full = returns_full[stable_cols]
        n_assets = returns_full.shape[1]
        print(
            f"  Fig 40: rolling HMM-blended moments, "
            f"{n_assets} assets, {len(returns_full)} days"
        )

        hmm_cfg = HMMConfig(n_states=2, random_state=42)

        roll_dates: list[pd.Timestamp] = []
        hmm_mu_vals: list[float] = []
        emp_mu_vals: list[float] = []
        hmm_vol_vals: list[float] = []
        emp_vol_vals: list[float] = []

        n_rows = len(returns_full)
        indices = list(range(_WINDOW_DAYS, n_rows, _STEP))
        if not indices:
            # Fallback: single window at end
            indices = [n_rows]

        for end_idx in indices:
            start_idx = max(0, end_idx - _WINDOW_DAYS)
            window = returns_full.iloc[start_idx:end_idx]
            if len(window) < 60:
                continue

            roll_dates.append(returns_full.index[end_idx - 1])

            # ── HMM-blended mu ──────────────────────────────────────────────
            try:
                hmm_mu_est = HMMBlendedMu(hmm_config=hmm_cfg)
                hmm_mu_est.fit(window)
                hmm_mu_mean = float(np.mean(hmm_mu_est.mu_)) * _TRADING_DAYS
            except Exception:
                hmm_mu_mean = float("nan")
            hmm_mu_vals.append(hmm_mu_mean)

            # ── Empirical mu ────────────────────────────────────────────────
            emp_mu_est = EmpiricalMu()
            emp_mu_est.fit(window)
            emp_mu_mean = float(np.mean(emp_mu_est.mu_)) * _TRADING_DAYS
            emp_mu_vals.append(emp_mu_mean)

            # ── HMM-blended covariance → annualised vol ─────────────────────
            try:
                hmm_cov_est = HMMBlendedCovariance(hmm_config=hmm_cfg)
                hmm_cov_est.fit(window)
                hmm_vol = float(
                    np.sqrt(np.mean(np.diag(hmm_cov_est.covariance_)) * _TRADING_DAYS)
                )
            except Exception:
                hmm_vol = float("nan")
            hmm_vol_vals.append(hmm_vol)

            # ── Empirical covariance → annualised vol ───────────────────────
            emp_cov_est = EmpiricalCovariance()
            emp_cov_est.fit(window)
            emp_vol = float(
                np.sqrt(np.mean(np.diag(emp_cov_est.covariance_)) * _TRADING_DAYS)
            )
            emp_vol_vals.append(emp_vol)

        if not roll_dates:
            print("  Fig 40: insufficient data for rolling windows — skipped")
            return

        dates = pd.DatetimeIndex(roll_dates)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # ── Panel 1: expected return ─────────────────────────────────────────
        ax1.plot(
            dates,
            np.array(hmm_mu_vals) * 100,
            color="#1565C0",
            linewidth=1.5,
            label="HMM-blended μ (full law of total variance)",
        )
        ax1.plot(
            dates,
            np.array(emp_mu_vals) * 100,
            color="#757575",
            linewidth=1.2,
            linestyle="--",
            label="Empirical μ",
        )
        ax1.axhline(0, color="#BDBDBD", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("Annualised expected return (%)")
        ax1.set_title(
            "Panel A: Rolling Annualised Expected Return\n"
            f"(2-year window, {n_assets} assets, monthly step)"
        )
        ax1.legend(loc="upper right", fontsize=9)

        # ── Panel 2: volatility ──────────────────────────────────────────────
        ax2.plot(
            dates,
            np.array(hmm_vol_vals) * 100,
            color="#C62828",
            linewidth=1.5,
            label="HMM-blended σ (incl. cross-state dispersion)",
        )
        ax2.plot(
            dates,
            np.array(emp_vol_vals) * 100,
            color="#757575",
            linewidth=1.2,
            linestyle="--",
            label="Empirical σ",
        )
        ax2.set_ylabel("Annualised volatility (%)")
        ax2.set_xlabel("Date")
        ax2.set_title("Panel B: Rolling Annualised Volatility (√mean diagonal covariance)")
        ax2.legend(loc="upper right", fontsize=9)

        fig.suptitle(
            "HMM-Blended vs Empirical Moment Estimates Over Time\n"
            "Warning: HMMBlendedCovariance ≠ blend_moments_by_regime "
            "(latter omits cross-state dispersion term)",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
