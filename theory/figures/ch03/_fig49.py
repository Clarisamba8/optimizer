"""Fig49TrackRecord — forecast vs realised returns for omega calibration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimizer.views._uncertainty import calibrate_omega_from_track_record
from theory.figures._base import FigureGenerator

_N_MONTHS = 24
_RNG_SEED = 42


class Fig49TrackRecord(FigureGenerator):
    """Scatter plot: historical view forecasts vs realised returns.

    Generates synthetic monthly view/return data with known properties,
    then calls calibrate_omega_from_track_record() to show actual omega
    computation. Shows regression line and forecast error variance.
    """

    @property
    def name(self) -> str:
        return "fig_49_track_record"

    def generate(self) -> None:
        rng = np.random.default_rng(_RNG_SEED)

        # Synthetic data: forecaster with moderate skill
        true_returns = rng.normal(0.008, 0.04, _N_MONTHS)  # monthly
        forecast_noise = rng.normal(0.0, 0.025, _N_MONTHS)
        forecasts = true_returns * 0.6 + 0.004 + forecast_noise  # some skill

        # Create DataFrames for the calibration function
        import pandas as pd
        dates = pd.date_range("2023-01-01", periods=_N_MONTHS, freq="ME")
        view_history = pd.DataFrame(
            {"view_1": forecasts}, index=dates,
        )
        return_history = pd.DataFrame(
            {"view_1": true_returns}, index=dates,
        )

        # Call the actual optimizer function
        omega = calibrate_omega_from_track_record(view_history, return_history)
        omega_val = omega[0, 0]

        error_std = np.sqrt(omega_val)

        print(f"  Fig 49: omega_11 = {omega_val:.6f}, error_std = {error_std:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter
        ax.scatter(
            forecasts * 100, true_returns * 100,
            c="#2196F3", s=60, alpha=0.7, edgecolors="white", linewidth=0.5,
            zorder=3,
        )

        # Regression line
        coeffs = np.polyfit(forecasts, true_returns, 1)
        x_line = np.linspace(forecasts.min(), forecasts.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line * 100, y_line * 100, color="#FF5722", lw=2,
                label=f"Regression (slope={coeffs[0]:.2f})")

        # Perfect calibration line
        lim_min = min(forecasts.min(), true_returns.min()) * 100
        lim_max = max(forecasts.max(), true_returns.max()) * 100
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color="#9E9E9E", ls="--", lw=1, label="Perfect calibration")

        # Annotate omega
        ax.text(
            0.05, 0.95,
            (
                f"Forecast error variance:\n"
                rf"$\omega_k$ = Var(Q - r) = {omega_val:.5f}"
                f"\n= ({error_std * 100:.2f}% monthly std)"
            ),
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E3F2FD", "alpha": 0.8},
        )

        ax.set_xlabel("Forecast Return (%)")
        ax.set_ylabel("Realised Return (%)")
        ax.set_title(
            "Calibrating View Uncertainty from Forecast Track Record\n"
            f"{_N_MONTHS} monthly observations | "
            r"$\omega_k = \mathrm{Var}(Q_k - r_k)$",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig)
