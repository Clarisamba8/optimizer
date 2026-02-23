"""Fig19ScalingComparison — naive linear scaling vs log-normal correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from optimizer.moments._scaling import scale_moments_to_horizon
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns


class Fig19ScalingComparison(FigureGenerator):
    """Line chart: naive linear scaling vs log-normal correction using real moments.

    Critical fix: the naive baseline uses
    ``mu_arith_daily = float(np.expm1(mu_log_daily + 0.5 * sigma_daily ** 2))``
    NOT ``mu_daily * T_days``, to avoid a degenerate flat line caused by
    log-return means cancelling cross-universe.
    """

    @property
    def name(self) -> str:
        return "fig_19_scaling_comparison"

    def generate(self) -> None:
        prices = self._prices

        # Clean arithmetic returns first (DataValidator + OutlierTreater) so that
        # data-error spikes (e.g. +900% days from price-feed errors) do not bias
        # the cross-universe average mu and sigma that parameterise the scaling curves.
        # Log returns are then derived from the cleaned arithmetic returns via
        # log(1 + r_arith), which is mathematically equivalent to log(P_t / P_{t-1}).
        arith_clean = clean_returns(prices_to_returns(prices)).dropna(how="all")
        log_rets = np.log1p(arith_clean)

        # Stack to 1D series (drops NaN automatically) then compute stats
        log_rets_flat = log_rets.stack()
        mu_log_daily = float(log_rets_flat.mean())
        sigma_daily = float(log_rets_flat.std())

        # Arithmetic (simple) daily mean: mu_arith = exp(mu_log + 0.5*sigma^2) - 1
        # This is what a naive practitioner observes and scales linearly.
        # Using mu_log directly gives ~0 for cross-universe averages (losers cancel
        # winners), producing a degenerate flat naive line and hiding the real bias.
        mu_arith_daily = float(np.expm1(mu_log_daily + 0.5 * sigma_daily ** 2))

        print(
            f"  Fig 19: avg daily mu_log={mu_log_daily:.6f}, "
            f"mu_arith={mu_arith_daily:.6f}, sigma={sigma_daily:.6f}  "
            f"(annualised: mu_log={mu_log_daily*252*100:.1f}%, "
            f"mu_arith={mu_arith_daily*252*100:.1f}%, "
            f"sigma={sigma_daily*np.sqrt(252)*100:.1f}%)"
        )

        tickers_single = ["UNIVERSE_AVG"]
        mu_s = pd.Series([mu_log_daily], index=tickers_single)
        cov_s = pd.DataFrame(
            [[sigma_daily ** 2]], index=tickers_single, columns=tickers_single
        )

        horizons = np.arange(1, 61)   # 1-60 months (~21 trading days each)

        naive_return, naive_vol, exact_return, exact_vol = [], [], [], []
        for T_months in horizons:
            T_days = T_months * 21
            # Naive: linearly scale arithmetic daily mean (what practitioners actually do)
            naive_return.append(mu_arith_daily * T_days * 100)
            naive_vol.append(sigma_daily * np.sqrt(T_days) * 100)
            mu_T, cov_T = scale_moments_to_horizon(mu_s, cov_s, T_days)
            exact_return.append(mu_T.iloc[0] * 100)
            exact_vol.append(np.sqrt(cov_T.iloc[0, 0]) * 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(horizons, naive_return, lw=2, color="#FF5722", ls="--",
                 label="Naive: mu_arith * T")
        ax1.plot(horizons, exact_return, lw=2, color="#2196F3", label="Log-normal: E[R_T]")
        ax1.fill_between(horizons, naive_return, exact_return, alpha=0.15, color="#9C27B0",
                         label="Convexity bias")
        ax1.axvline(12, color="grey", lw=1, ls=":", label="12-month mark")
        ax1.set_xlabel("Investment Horizon (months)")
        ax1.set_ylabel("Expected Cumulative Return (%)")
        ax1.set_title("Expected Return Scaling")
        ax1.legend(fontsize=9)

        ax2.plot(horizons, naive_vol, lw=2, color="#FF5722", ls="--", label="Naive: sigma * sqrt(T)")
        ax2.plot(horizons, exact_vol, lw=2, color="#2196F3", label="Log-normal: std[R_T]")
        ax2.fill_between(horizons, naive_vol, exact_vol, alpha=0.15, color="#9C27B0",
                         label="Compounding gap")
        ax2.axvline(12, color="grey", lw=1, ls=":", label="12-month mark")
        ax2.set_xlabel("Investment Horizon (months)")
        ax2.set_ylabel("Cumulative Volatility (%)")
        ax2.set_title("Volatility Scaling")
        ax2.legend(fontsize=9)

        ann_mu_arith = mu_arith_daily * 252 * 100
        ann_mu_log = mu_log_daily * 252 * 100
        ann_sig = sigma_daily * np.sqrt(252) * 100
        fig.suptitle(
            f"Naive vs Log-Normal Multi-Period Moment Scaling\n"
            f"(cross-universe avg: mu_arith={ann_mu_arith:.1f}%/yr, "
            f"mu_log={ann_mu_log:.1f}%/yr, sigma={ann_sig:.1f}%/yr - "
            f"divergence grows beyond ~12 months)",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig)
