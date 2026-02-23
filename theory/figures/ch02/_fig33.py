"""Fig33LWEigenvalues — raw vs Ledoit-Wolf covariance eigenvalues stem plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import EmpiricalCovariance, LedoitWolf

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 756
# Target N/T ≈ 0.3-0.5 to show visible LW compression while keeping N < T.
# With T ≈ 477 days, 200 assets gives N/T ≈ 0.42 — well-conditioned enough
# to be interesting but not so ill-conditioned that the raw spectrum diverges.
_MAX_ASSETS = 200


class Fig33LWEigenvalues(FigureGenerator):
    """Stem plot: raw sample covariance vs Ledoit-Wolf shrunk eigenvalues.

    Large top eigenvalues in the raw sample covariance indicate market-factor
    concentration; the smallest eigenvalues are often noise artefacts.
    Ledoit-Wolf shrinkage compresses the spread: large eigenvalues shrink down
    and small eigenvalues shrink up toward a common target, producing a more
    numerically stable and better-conditioned covariance matrix.

    Both covariances are fitted on the same 3-year window of clean returns.
    Eigenvalues are sorted descending and plotted on a log scale so the full
    dynamic range is visible.
    """

    @property
    def name(self) -> str:
        return "fig_33_lw_eigenvalues"

    def generate(self) -> None:
        prices = self._prices

        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        # Cap asset count
        returns = returns.iloc[:, :_MAX_ASSETS]
        n_assets = returns.shape[1]
        print(f"  Fig 33: eigenvalues for {n_assets} assets x {len(returns)} days")

        emp_cov = EmpiricalCovariance()
        emp_cov.fit(returns)
        eig_raw = np.sort(np.linalg.eigvalsh(emp_cov.covariance_).real)[::-1]

        lw = LedoitWolf()
        lw.fit(returns)
        eig_lw = np.sort(np.linalg.eigvalsh(lw.covariance_).real)[::-1]

        indices = np.arange(1, n_assets + 1)

        # For the linear panel zoom into the top 20 eigenvalues where the
        # LW compression is clearly visible.  The full spectrum is dominated
        # by the market-factor eigenvalue and everything else is invisible.
        _TOP_K = 20
        top_idx = indices[:_TOP_K]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # --- Linear scale: top-K eigenvalues only ---
        ax1.plot(top_idx, eig_raw[:_TOP_K], "o-", color="#2196F3", lw=1.8,
                 markersize=7, label="Raw sample")
        ax1.plot(top_idx, eig_lw[:_TOP_K],  "s-", color="#FF5722", lw=1.8,
                 markersize=7, label="Ledoit-Wolf")
        ax1.fill_between(top_idx, eig_raw[:_TOP_K], eig_lw[:_TOP_K],
                         alpha=0.15, color="#9C27B0",
                         label="Shrinkage gap")
        ax1.set_xlabel("Eigenvalue rank")
        ax1.set_ylabel("Eigenvalue (linear scale)")
        ax1.set_title(f"Top {_TOP_K} Eigenvalues — Linear Scale\n"
                      "(LW shrinks large values down)")
        ax1.legend(fontsize=9)

        # --- Log scale (reveals small-eigenvalue noise floor) ---
        eig_raw_pos = np.maximum(eig_raw, 1e-10)
        eig_lw_pos  = np.maximum(eig_lw,  1e-10)
        ax2.semilogy(indices, eig_raw_pos, "o-", color="#2196F3", lw=1.5,
                     markersize=5, label="Raw sample")
        ax2.semilogy(indices, eig_lw_pos,  "s-", color="#FF5722", lw=1.5,
                     markersize=5, label="Ledoit-Wolf")
        ax2.set_xlabel("Eigenvalue rank")
        ax2.set_ylabel("Eigenvalue (log scale)")
        ax2.set_title("Eigenvalue Spectrum — Log Scale")
        ax2.legend(fontsize=9)
        ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)

        fig.suptitle(
            "Ledoit-Wolf Shrinkage: Eigenvalue Compression\n"
            f"(3-year window, {n_assets} assets — LW shrinks large values down, "
            "small values up)",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
