"""Fig34MarchenkoPastur — empirical eigenvalue histogram with MP PDF overlay."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import EmpiricalCovariance

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 756


def _marchenko_pastur_pdf(
    lam: np.ndarray,
    q: float,
    sigma_sq: float = 1.0,
) -> np.ndarray:
    """Marchenko-Pastur probability density function.

    Parameters
    ----------
    lam:
        Array of eigenvalue values at which to evaluate the PDF.
    q:
        Ratio N/T (number of assets / number of time periods).
    sigma_sq:
        Variance of the underlying population (1.0 for correlation matrices).

    Returns
    -------
    np.ndarray
        PDF values; zero outside the support [λ₋, λ₊].

    Notes
    -----
    The density is:

    .. math::

        \\rho(\\lambda) = \\frac{
            \\sqrt{(\\lambda_+ - \\lambda)(\\lambda - \\lambda_-)}
        }{2 \\pi N q \\sigma^2 \\lambda}

    for :math:`\\lambda \\in [\\lambda_-, \\lambda_+]`, where:

    .. math::

        \\lambda_{\\pm} = \\sigma^2 (1 \\pm \\sqrt{q})^2
    """
    lam_plus  = sigma_sq * (1.0 + np.sqrt(q)) ** 2
    lam_minus = sigma_sq * (1.0 - np.sqrt(q)) ** 2

    pdf = np.zeros_like(lam, dtype=float)
    mask = (lam >= lam_minus) & (lam <= lam_plus)
    lam_m = lam[mask]

    under_root = np.maximum((lam_plus - lam_m) * (lam_m - lam_minus), 0.0)
    # Formula: 1 / (2*pi*sigma^2*q) * sqrt((lam_+ - lam)(lam - lam_-)) / lam
    pdf[mask] = (1.0 / (2.0 * np.pi * sigma_sq * q)) * (
        np.sqrt(under_root) / lam_m
    )
    return pdf


class Fig34MarchenkoPastur(FigureGenerator):
    """Histogram of empirical correlation-matrix eigenvalues with MP PDF overlay.

    The Marchenko-Pastur law describes the limiting spectral distribution of
    a random Wishart matrix as T, N → ∞ with q = N/T fixed.  Eigenvalues that
    fall inside the MP bulk [λ₋, λ₊] are consistent with pure noise; eigenvalues
    above λ₊ carry genuine signal (market factors, sector factors, etc.).

    The figure uses the sample *correlation* matrix (unit-variance normalised)
    so that the MP parameterisation with σ² = 1 applies directly.
    """

    @property
    def name(self) -> str:
        return "fig_34_marchenko_pastur"

    def generate(self) -> None:
        prices = self._prices

        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        n_obs = len(returns)
        # Marchenko-Pastur law requires N < T (q = N/T < 1) for the standard
        # single-MP-bulk picture.  When N > T the sample covariance is singular
        # and the bulk structure is obscured by a mass of zero eigenvalues.
        # Target q ≈ 0.5 for a pedagogically clear plot.
        n_target = max(10, int(n_obs * 0.5))
        returns = returns.iloc[:, :n_target]

        n_obs, n_assets = returns.shape
        q = n_assets / n_obs  # concentration ratio N/T
        print(f"  Fig 34: T={n_obs}, N={n_assets}, q={q:.3f}")

        # Sample correlation matrix
        emp_cov = EmpiricalCovariance()
        emp_cov.fit(returns)
        cov = emp_cov.covariance_

        std_diag = np.sqrt(np.diag(cov))
        # Guard against zero-variance columns (clean_returns already handles this
        # but be defensive)
        std_diag = np.where(std_diag < 1e-12, 1.0, std_diag)
        corr = cov / np.outer(std_diag, std_diag)
        np.fill_diagonal(corr, 1.0)

        eigenvalues = np.linalg.eigvalsh(corr).real  # sorted ascending

        sigma_sq = 1.0  # correlation matrix convention
        lam_plus  = sigma_sq * (1.0 + np.sqrt(q)) ** 2
        lam_minus = sigma_sq * (1.0 - np.sqrt(q)) ** 2

        n_signal = int((eigenvalues > lam_plus).sum())
        n_noise  = n_assets - n_signal
        print(
            f"  MP lam+ = {lam_plus:.3f}, lam- = {lam_minus:.3f} | "
            f"signal eigenvalues: {n_signal}, noise: {n_noise}"
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram — focus on the bulk; extreme signal eigenvalues are cut off
        # to keep the MP curve visible.
        hist_max = lam_plus * 4.0
        plot_eigs = eigenvalues[eigenvalues <= hist_max]
        n_cut = int((eigenvalues > hist_max).sum())
        ax.hist(
            plot_eigs, bins=40, density=True,
            color="#2196F3", alpha=0.65, edgecolor="white", lw=0.5,
            label=f"Empirical ({n_assets} eigenvalues, {n_cut} cut off)",
        )

        # MP PDF overlay
        lam_grid = np.linspace(max(lam_minus * 0.5, 1e-4), lam_plus * 1.05, 500)
        mp_pdf = _marchenko_pastur_pdf(lam_grid, q, sigma_sq)
        ax.plot(lam_grid, mp_pdf, color="#F44336", lw=2.5, label="Marchenko-Pastur PDF")

        # lam_plus marker
        ax.axvline(
            lam_plus, color="#FF9800", lw=2.0, ls="--",
            label=f"lam+ = {lam_plus:.3f}  ({n_signal} signal eigenvector(s) above)",
        )
        ax.axvline(
            lam_minus, color="#9C27B0", lw=1.5, ls=":",
            label=f"lam- = {lam_minus:.3f}",
        )

        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_title(
            "Marchenko-Pastur Law: Noise vs Signal in the Correlation Matrix\n"
            f"(T={n_obs} days, N={n_assets} assets, q=N/T={q:.3f} — "
            f"{n_signal} signal eigenvalue(s) above lam+)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig)
