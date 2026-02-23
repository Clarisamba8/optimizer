"""Fig36Detoning — side-by-side heatmaps of full vs detoned correlation matrix."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skfolio.moments import DetoneCovariance, EmpiricalCovariance

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns

_WINDOW_DAYS = 756
_MAX_ASSETS = 30


class Fig36Detoning(FigureGenerator):
    """Side-by-side correlation matrix heatmaps: full vs detoned.

    The full correlation matrix is dominated by a market factor (the largest
    eigenvector) that inflates all pairwise correlations uniformly.  Detoning
    removes the contribution of the leading market eigencomponent, revealing
    the residual pairwise structure (sector effects, style tilts) that is
    otherwise hidden.

    Both matrices are derived from the same 3-year window of clean returns.
    Assets are reordered by the first eigenvector loading to visually group
    assets with similar market-factor exposure.
    """

    @property
    def name(self) -> str:
        return "fig_36_detoning"

    def generate(self) -> None:
        prices = self._prices

        p_window = prices.iloc[-_WINDOW_DAYS:] if len(prices) > _WINDOW_DAYS else prices
        returns = clean_returns(prices_to_returns(p_window.ffill()).dropna()).dropna()

        # Cap asset count for readability
        returns = returns.iloc[:, :_MAX_ASSETS]
        n_assets = returns.shape[1]
        tickers = returns.columns.tolist()
        short_names = [
            t.replace("_US_EQ", "").replace("_EQ", "") for t in tickers
        ]
        print(f"  Fig 36: {n_assets} assets x {len(returns)} days")

        # Full empirical covariance -> correlation
        emp = EmpiricalCovariance()
        emp.fit(returns)
        cov_full = emp.covariance_

        std = np.sqrt(np.diag(cov_full))
        std = np.where(std < 1e-12, 1.0, std)
        corr_full = cov_full / np.outer(std, std)
        np.fill_diagonal(corr_full, 1.0)

        # Detoned covariance -> correlation
        det = DetoneCovariance(n_markets=1)
        det.fit(returns)
        cov_det = det.covariance_

        std_det = np.sqrt(np.diag(cov_det))
        std_det = np.where(std_det < 1e-12, 1.0, std_det)
        corr_det = cov_det / np.outer(std_det, std_det)
        np.fill_diagonal(corr_det, 1.0)

        # Reorder assets by leading eigenvector loading so the block structure
        # in the detoned matrix becomes apparent
        _eigvals, eigvecs = np.linalg.eigh(corr_full)
        market_vec = eigvecs[:, -1].real  # largest eigenvalue
        order = np.argsort(market_vec)[::-1]

        corr_full_ord = corr_full[np.ix_(order, order)]
        corr_det_ord  = corr_det[np.ix_(order, order)]
        names_ord = [short_names[i] for i in order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        _plot_heatmap(ax1, corr_full_ord, names_ord, "Full Correlation Matrix")
        _plot_heatmap(
            ax2, corr_det_ord, names_ord,
            "Detoned Correlation Matrix\n(market factor removed)",
        )

        fig.suptitle(
            "Detoning: Removing the Market Factor\n"
            f"(3-year window, {n_assets} assets, sorted by market-factor loading)",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)


def _plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    """Render a correlation-matrix heatmap on the given axes.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    matrix:
        Square correlation matrix (values in [-1, 1]).
    labels:
        Asset tick labels (must match matrix dimensions).
    title:
        Axes title string.
    """
    n = len(labels)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_step = max(1, n // 20)
    tick_positions = list(range(0, n, tick_step))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(
        [labels[i] for i in tick_positions], rotation=45, ha="right", fontsize=7
    )
    ax.set_yticklabels([labels[i] for i in tick_positions], fontsize=7)
    ax.set_title(title, fontsize=10)
