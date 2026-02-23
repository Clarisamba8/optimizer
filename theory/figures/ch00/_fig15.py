"""Fig15ISvsOOSIC — In-sample vs out-of-sample IC stability across CV folds."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer.factors._config import FactorConstructionConfig
from optimizer.factors._construction import compute_all_factors
from optimizer.factors._oos_validation import FactorOOSConfig, run_factor_oos_validation
from optimizer.factors._standardization import standardize_all_factors
from optimizer.factors._validation import compute_ic_series, run_factor_validation
from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_returns
from theory.figures.ch00._data_helpers import query_fundamentals, query_sector_labels

_N_PANELS = 5
_IS_COLOR = "#1565C0"
_OOS_COLOR = "#FF9800"


class Fig15ISvsOOSIC(FigureGenerator):
    """Multi-panel chart comparing IS and OOS IC across rolling CV folds.

    Parameters
    ----------
    prices:
        Wide price DataFrame.
    output_dir:
        Directory where the generated PNG is saved.
    db_url:
        PostgreSQL connection string for querying fundamentals and sectors.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        output_dir: Path,
        db_url: str,
    ) -> None:
        super().__init__(prices, output_dir)
        self._db_url = db_url

    @property
    def name(self) -> str:
        return "fig_15_is_vs_oos_ic"

    def generate(self) -> None:
        prices = self._prices.ffill()
        returns = clean_returns(prices_to_returns(prices)).dropna()
        tickers = returns.columns.tolist()
        print(f"  Fig 15: {len(tickers)} assets")

        fundamentals = query_fundamentals(self._db_url, tickers)
        sectors = query_sector_labels(self._db_url, tickers)

        if fundamentals.empty or len(fundamentals) < 20:
            print("  Fig 15: insufficient fundamentals, using fallback.")
            self._generate_fallback()
            return

        try:
            config = FactorConstructionConfig.for_all_factors()
            raw_factors = compute_all_factors(
                fundamentals, prices, config=config,
            )
            std_factors, _ = standardize_all_factors(
                raw_factors,
                sector_labels=sectors if not sectors.empty else None,
            )
        except Exception as exc:
            print(f"  Fig 15: factor computation failed ({exc}), using fallback.")
            self._generate_fallback()
            return

        # Build monthly score and return histories
        monthly_prices = prices.resample("ME").last().dropna(how="all")
        monthly_ret = monthly_prices.pct_change().dropna(how="all")

        available = [
            c for c in std_factors.columns if std_factors[c].notna().sum() >= 20
        ]
        if len(available) < 3:
            print("  Fig 15: too few valid factors, using fallback.")
            self._generate_fallback()
            return

        common_tickers = std_factors.index.intersection(monthly_ret.columns)
        if len(common_tickers) < 20:
            print("  Fig 15: too few common tickers, using fallback.")
            self._generate_fallback()
            return

        # Build factor_scores_history for IS validation
        factor_scores_history: dict[str, pd.DataFrame] = {}
        for factor_name in available:
            scores_s = std_factors[factor_name].reindex(common_tickers).dropna()
            if len(scores_s) < 20:
                continue
            # Repeat cross-section across time
            scores_matrix = pd.DataFrame(
                dict.fromkeys(monthly_ret.index, scores_s),
            ).T
            scores_matrix.index = monthly_ret.index
            scores_matrix = scores_matrix[scores_s.index]
            factor_scores_history[factor_name] = scores_matrix

        returns_history = monthly_ret[common_tickers]

        # IS validation to pick top factors by |mean_ic|
        report = run_factor_validation(factor_scores_history, returns_history)
        ic_by_factor = {
            r.factor_name: abs(r.mean_ic) for r in report.ic_results
        }
        top_factors = sorted(ic_by_factor, key=ic_by_factor.get, reverse=True)[  # type: ignore[arg-type]
            :_N_PANELS
        ]

        if len(top_factors) < 2:
            print("  Fig 15: too few IC results, using fallback.")
            self._generate_fallback()
            return

        # Build MultiIndex panel for OOS validation
        rows = []
        for date in monthly_ret.index:
            for ticker in common_tickers:
                row: dict[str, float] = {}
                for factor_name in top_factors:
                    if factor_name in factor_scores_history:
                        mat = factor_scores_history[factor_name]
                        if date in mat.index and ticker in mat.columns:
                            row[factor_name] = mat.loc[date, ticker]
                rows.append({"date": date, "ticker": ticker, **row})

        panel_df = pd.DataFrame(rows)
        panel_df = panel_df.set_index(["date", "ticker"])
        scores_mi = panel_df[top_factors]

        returns_flat = monthly_ret[common_tickers].stack()
        returns_flat.index.names = ["date", "ticker"]
        returns_mi = returns_flat.to_frame("return")

        # Run OOS validation
        oos_config = FactorOOSConfig(train_months=24, val_months=6, step_months=6)
        try:
            oos_result = run_factor_oos_validation(
                scores_mi, returns_mi, config=oos_config,
            )
        except Exception as exc:
            print(f"  Fig 15: OOS validation failed ({exc}), using fallback.")
            self._generate_fallback()
            return

        if oos_result.n_folds < 2:
            print("  Fig 15: too few OOS folds, using fallback.")
            self._generate_fallback()
            return

        # Compute IS IC per fold for comparison
        all_dates = scores_mi.index.get_level_values(0).unique().sort_values()
        n_total = len(all_dates)
        is_ic_rows: list[dict[str, float]] = []
        folds_used = min(oos_result.n_folds, len(oos_result.per_fold_ic))

        for i in range(folds_used):
            t_start = i * oos_config.step_months
            t_end = t_start + oos_config.train_months
            if t_end > n_total:
                break
            train_dates = all_dates[t_start:t_end]
            fold_is: dict[str, float] = {}
            for factor_name in top_factors:
                if factor_name not in factor_scores_history:
                    fold_is[factor_name] = float("nan")
                    continue
                mat = factor_scores_history[factor_name]
                train_mat = mat.loc[mat.index.isin(train_dates)]
                train_ret = returns_history.loc[returns_history.index.isin(train_dates)]
                ic_s = compute_ic_series(train_mat, train_ret, factor_name)
                fold_is[factor_name] = (
                    float(ic_s.mean()) if len(ic_s) > 0 else float("nan")
                )
            is_ic_rows.append(fold_is)

        is_ic_df = pd.DataFrame(is_ic_rows, columns=top_factors)
        self._plot(top_factors, is_ic_df, oos_result.per_fold_ic[top_factors])

    def _generate_fallback(self) -> None:
        """Synthetic IS vs OOS IC for illustration."""
        rng = np.random.default_rng(42)
        factor_names = [
            "momentum_12_1", "book_to_price", "gross_profitability",
            "roe", "volatility",
        ]
        n_folds = 8
        is_data: dict[str, list[float]] = {}
        oos_data: dict[str, list[float]] = {}
        for f in factor_names:
            base = rng.uniform(0.02, 0.08)
            is_data[f] = (rng.normal(base, 0.02, n_folds)).tolist()
            decay = rng.uniform(0.5, 0.9)
            oos_data[f] = (rng.normal(base * decay, 0.03, n_folds)).tolist()

        is_df = pd.DataFrame(is_data)
        oos_df = pd.DataFrame(oos_data)
        self._plot(factor_names, is_df, oos_df)

    def _plot(
        self,
        factor_names: list[str],
        is_ic: pd.DataFrame,
        oos_ic: pd.DataFrame,
    ) -> None:
        n = min(len(factor_names), _N_PANELS)
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
        axes_flat = axes.flatten()

        for i in range(n):
            ax = axes_flat[i]
            fname = factor_names[i]
            display = fname.replace("_", " ").title()

            folds = np.arange(1, len(is_ic) + 1)
            is_vals = (
                is_ic[fname].values
                if fname in is_ic.columns
                else np.zeros(len(folds))
            )
            oos_vals = (
                oos_ic[fname].values[:len(folds)]
                if fname in oos_ic.columns
                else np.zeros(len(folds))
            )

            # Align lengths
            min_len = min(len(folds), len(is_vals), len(oos_vals))
            folds = folds[:min_len]
            is_vals = is_vals[:min_len]
            oos_vals = oos_vals[:min_len]

            ax.plot(folds, is_vals, "o-", color=_IS_COLOR, lw=1.8, ms=5, label="IS IC")
            ax.plot(
                folds, oos_vals, "s--", color=_OOS_COLOR, lw=1.8, ms=5, label="OOS IC",
            )
            ax.axhline(0, color="black", lw=0.6, ls="-", alpha=0.4)

            is_mean = float(np.nanmean(is_vals))
            oos_mean = float(np.nanmean(oos_vals))
            decay_ratio = oos_mean / is_mean if abs(is_mean) > 1e-6 else float("nan")

            ax.set_title(
                f"{display}\nDecay: {decay_ratio:.0%}" if not np.isnan(decay_ratio)
                else display,
                fontsize=10,
            )
            ax.set_xlabel("Fold", fontsize=8)
            ax.set_ylabel("Mean IC", fontsize=8)
            ax.legend(fontsize=7, loc="best")
            ax.tick_params(labelsize=8)

        # Hide unused panels
        for j in range(n, len(axes_flat)):
            # Summary panel in last slot
            if j == n:
                ax = axes_flat[j]
                rows_data = []
                for fname in factor_names[:n]:
                    display = fname.replace("_", " ").title()
                    is_mean = (
                        float(is_ic[fname].mean())
                        if fname in is_ic.columns else 0.0
                    )
                    oos_mean = (
                        float(oos_ic[fname].mean())
                        if fname in oos_ic.columns else 0.0
                    )
                    ratio = (
                        oos_mean / is_mean
                        if abs(is_mean) > 1e-6
                        else float("nan")
                    )
                    rows_data.append([
                        display[:18],
                        f"{is_mean:.3f}",
                        f"{oos_mean:.3f}",
                        f"{ratio:.0%}",
                    ])

                ax.axis("off")
                table = ax.table(
                    cellText=rows_data,
                    colLabels=["Factor", "IS IC", "OOS IC", "Ratio"],
                    loc="center",
                    cellLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.4)
                ax.set_title("Summary", fontsize=10)
            else:
                axes_flat[j].set_visible(False)

        fig.suptitle(
            "In-Sample vs Out-of-Sample IC Stability Across CV Folds",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()
        self._save(fig)
