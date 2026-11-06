"""All report figures. Each function reads saved results and writes one PNG."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import config as cfg
from .evaluate import to_band

# Colour follows the entity and stays the same in every figure.
TUNED, BASELINE, RANDOM = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACTUAL = "#3a3a38"
BAND_NAMES = [label for _, label in cfg.AQI_BANDS]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "text.color": INK, "legend.frameon": False, "lines.linewidth": 1.8,
})


def _save(fig, name):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES_DIR / name, facecolor="white")
    plt.close(fig)


def data_overview(raw, ds):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(raw.index, raw["AQI"], color=ACTUAL, linewidth=0.9)
    start = pd.Timestamp(cfg.TEST_START)
    ax.axvspan(start, raw.index.max(), color=TUNED, alpha=0.08, linewidth=0)
    ax.axvline(start, color=TUNED, linewidth=1.2)
    ax.text(start + pd.Timedelta(days=12), raw["AQI"].max() * 0.97, "Held-out test period", color=MUTED, va="top")
    ax.text(raw.index.min() + pd.Timedelta(days=12), raw["AQI"].max() * 0.97, "Training period", color=MUTED, va="top")
    ax.set(title=f"Daily AQI, {ds.report['city']}, with the chronological split", ylabel="AQI")
    _save(fig, "01_aqi_timeline.png")


def missing_values(report):
    s = pd.Series(report["missing_pct"]).sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(s.index, s.values, color=ACTUAL, height=0.65)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color=MUTED, fontsize=9)
    ax.set(title="Missing values by column, before cleaning", xlabel="Days missing (%)", xlim=(0, max(s.max() * 1.4, 1)))
    ax.grid(axis="y", visible=False)
    _save(fig, "02_missing_values.png")


def correlations(ds):
    frame = ds.X_train.assign(AQI_next_day=ds.y_train)
    cols = ["AQI_next_day", "AQI", "AQI_roll_mean_7", "AQI_roll_mean_30", "temp_min", "dew_point", "pressure",
            "rain", "humidity", "wind_mean", "wind_max", "radiation"]
    corr = frame[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.4, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.65 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman correlation")
    ax.set_title("Feature correlations in the training period")
    _save(fig, "03_correlations.png")


def convergence(runs, baseline_cv):
    pso = [r for r in runs if r["method"] == "PSO"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    curves = np.array([r["gbest_history"] for r in pso])
    x = np.arange(1, curves.shape[1] + 1)
    for curve in curves:
        ax.plot(x, curve, color=TUNED, alpha=0.3, linewidth=1)
    ax.plot(x, curves.mean(axis=0), color=TUNED, label=f"PSO global best, mean of {len(pso)} runs")
    ax.axhline(baseline_cv, color=BASELINE, linestyle="--", label="Default Random Forest")
    ax.set(title="PSO convergence on walk-forward validation RMSE", xlabel="Iteration", ylabel="Validation RMSE (AQI points)")
    ax.legend(loc="center right")
    _save(fig, "04_pso_convergence.png")


def pso_vs_random(runs, baseline_cv):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for method, color in (("PSO", TUNED), ("Random search", RANDOM)):
        curves = np.array([r["best_by_evaluation"] for r in runs if r["method"] == method])
        x = np.arange(1, curves.shape[1] + 1)
        ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.plot(x, curves.mean(axis=0), color=color, label=f"{method}, mean and range")
    hp_only = [r["best_by_evaluation"] for r in runs if r["method"] == "PSO (hyperparameters only)"]
    if hp_only:
        curves = np.array(hp_only)
        ax.plot(np.arange(1, curves.shape[1] + 1), curves.mean(axis=0), color=TUNED, linestyle=":",
                label="PSO, hyperparameters only, mean")
    ax.axhline(baseline_cv, color=BASELINE, linestyle="--", label="Default Random Forest")
    ax.set(title="Best validation RMSE found for the same evaluation budget", xlabel="Model evaluations", ylabel="Validation RMSE (AQI points)")
    ax.legend(loc="upper right")
    _save(fig, "05_pso_vs_random_search.png")


def swarm_dynamics(best_run):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(1, len(best_run["diversity_history"]) + 1)
    for ax, key, title in zip(axes, ("mean_history", "diversity_history", "inertia_history"),
                              ("Swarm mean RMSE", "Swarm diversity", "Inertia weight w")):
        ax.plot(x, best_run[key], color=TUNED)
        ax.set(title=title, xlabel="Iteration")
    fig.tight_layout()
    _save(fig, "06_swarm_dynamics.png")


def parameter_paths(best_run):
    positions = np.array(best_run["positions_history"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    iters = np.arange(1, positions.shape[0] + 1)
    for d, (ax, p) in enumerate(zip(axes.flat, cfg.SEARCH_SPACE)):
        values = p.low + positions[:, :, d] * (p.high - p.low)
        for i in range(values.shape[1]):
            ax.plot(iters, values[:, i], color=TUNED, alpha=0.3, linewidth=0.8)
        ax.axhline(best_run["params"][p.name], color=INK, linestyle="--", linewidth=1.1)
        ax.set_title(p.name)
    for ax in axes[1]:
        ax.set_xlabel("Iteration")
    fig.suptitle("Particle movement through each hyperparameter (dashed line: best value found)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_parameter_paths.png")


def metric_comparison(table):
    groups = [("Error, lower is better", ["RMSE", "MAE"]), ("Fit, category and alert scores, higher is better", ["Accuracy", "F1_macro", "R2", "AUC_high_pollution"])]
    models = [("Persistence", ACTUAL), ("Default RF", BASELINE), ("PSO-tuned RF", TUNED)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.7]})
    for ax, (title, metrics) in zip(axes, groups):
        width = 0.26
        for k, (name, color) in enumerate(models):
            vals = table.loc[name, metrics].astype(float).values
            bars = ax.bar(np.arange(len(metrics)) + (k - 1) * width, vals, width * 0.92, color=color, label=name)
            ax.bar_label(bars, fmt="%.1f" if max(vals) > 2 else "%.3f", fontsize=7.5, padding=2, color=MUTED)
        ax.set_xticks(np.arange(len(metrics)), [m.replace("_", " ") for m in metrics])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[1].set_ylim(0, 1.12)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "08_metric_comparison.png")


def forecast_timeline(pred):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4))
    zoom = pred.loc["2024-10-15":"2025-01-31"]
    for ax, frame, title in ((axes[0], pred, "Whole test period, January 2024 to October 2026"), (axes[1], zoom, "Winter smog season, October 2024 to January 2025")):
        ax.plot(frame.index, frame["actual"], color=ACTUAL, linewidth=1.3, label="Observed AQI")
        ax.plot(frame.index, frame["baseline"], color=BASELINE, linewidth=1, alpha=0.85, label="Default RF")
        ax.plot(frame.index, frame["tuned"], color=TUNED, linewidth=1.3, label="PSO-tuned RF")
        ax.set(title=title, ylabel="AQI")
    axes[0].legend(loc="upper right", ncols=3)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    _save(fig, "09_forecast_timeline.png")


def actual_vs_predicted(pred):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharex=True, sharey=True)
    lim = [0, max(pred["actual"].max(), pred["tuned"].max()) * 1.05]
    for ax, col, color, name in ((axes[0], "baseline", BASELINE, "Default RF"), (axes[1], "tuned", TUNED, "PSO-tuned RF")):
        ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1)
        ax.scatter(pred["actual"], pred[col], s=12, color=color, alpha=0.5, linewidths=0)
        ax.set(title=name, xlabel="Observed AQI", xlim=lim, ylim=lim, aspect="equal")
    axes[0].set_ylabel("Predicted AQI")
    fig.tight_layout()
    _save(fig, "10_actual_vs_predicted.png")


def residuals(pred):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    err = pred["actual"] - pred["tuned"]
    axes[0].hist(err, bins=40, color=TUNED, edgecolor="white", linewidth=0.4)
    axes[0].axvline(0, color=INK, linewidth=1)
    axes[0].set(title="PSO-tuned RF error distribution", xlabel="Observed minus predicted AQI", ylabel="Days")
    axes[0].grid(axis="x", visible=False)
    monthly = np.abs(err).groupby(err.index.month).mean()
    names = pd.to_datetime(monthly.index.astype(str), format="%m").strftime("%b")
    axes[1].bar(names, monthly.values, color=TUNED, width=0.68)
    axes[1].set(title="Mean absolute error by calendar month", ylabel="MAE (AQI points)")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "11_residuals.png")


def learning_curves(curves):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True, sharey=True)
    for ax, key, color, title in ((axes[0], "baseline", BASELINE, "Default RF"), (axes[1], "tuned", TUNED, "PSO-tuned RF")):
        c = curves[key]
        ax.plot(c["sizes"], c["train"], color=color, linestyle=":", label="Training RMSE")
        ax.plot(c["sizes"], c["val"], color=color, marker="o", markersize=5, label="Validation RMSE")
        ax.set(title=title, xlabel="Training days")
        ax.legend(loc="center right")
    axes[0].set_ylabel("RMSE (AQI points)")
    fig.suptitle("Learning curves on a fixed validation window (the last 20% of the training period)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "12_learning_curves.png")


def feature_importance(importance):
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(top.index, top.values, color=TUNED, height=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=MUTED, fontsize=8.5)
    ax.set(title="Most important features, PSO-tuned RF", xlabel="Share of impurity reduction")
    ax.grid(axis="y", visible=False)
    _save(fig, "13_feature_importance.png")


def confusion(pred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, name in ((axes[0], "baseline", "Default RF"), (axes[1], "tuned", "PSO-tuned RF")):
        m = confusion_matrix(to_band(pred["actual"]), to_band(pred[col]), labels=BAND_NAMES)
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(6), BAND_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(6), BAND_NAMES)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=9, color="white" if m[i, j] > m.max() * 0.55 else INK)
        ax.set(title=name, xlabel="Predicted category", ylabel="Observed category")
    fig.tight_layout()
    _save(fig, "14_confusion_matrices.png")


def model_comparison(table):
    t = table.sort_values("RMSE", ascending=False)
    colors = [TUNED if n == "PSO-tuned RF" else BASELINE if n == "Default RF" else ACTUAL for n in t.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(t.index, t["RMSE"], color=colors, height=0.62)
    ax.bar_label(bars, fmt="%.2f", padding=3, color=MUTED, fontsize=9)
    ax.set(title="Test RMSE of every model compared", xlabel="RMSE (AQI points), lower is better", xlim=(0, t["RMSE"].max() * 1.15))
    ax.grid(axis="y", visible=False)
    _save(fig, "15_model_comparison.png")


def benchmark_convergence(bench):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (name, runs) in zip(axes, bench.items()):
        for curve in runs["curves"]:
            ax.plot(np.arange(1, len(curve) + 1), np.maximum(curve, 1e-12), color=TUNED, alpha=0.45, linewidth=1)
        ax.set(title=name, xlabel="Iteration", yscale="log")
    axes[0].set_ylabel("Best f(x), log scale")
    fig.tight_layout()
    _save(fig, "16_benchmark_convergence.png")


def feature_selection(frequency, selected, best_run):
    """Which features PSO kept, and how the swarm's feature count evolved."""
    freq = frequency.sort_values()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 6.8), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = [TUNED if f in selected else GRID for f in freq.index]
    ax.barh(freq.index, freq.values * 100, color=colors, height=0.7)
    ax.set(title="Share of PSO runs that kept each feature", xlabel="PSO runs (%)", xlim=(0, 105))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.text(102, -1.2, "blue: in the final model", color=MUTED, ha="right", fontsize=8)

    n_hp = len(cfg.SEARCH_SPACE)
    positions = np.array(best_run["positions_history"])[:, :, n_hp:]
    counts = (positions >= cfg.FEATURE_THRESHOLD).sum(axis=2)
    iters = np.arange(1, counts.shape[0] + 1)
    ax2.fill_between(iters, counts.min(axis=1), counts.max(axis=1), color=TUNED, alpha=0.15, linewidth=0)
    ax2.plot(iters, counts.mean(axis=1), color=TUNED, label="Swarm mean and range")
    ax2.axhline(len(selected), color=INK, linestyle="--", linewidth=1.1, label=f"Final model: {len(selected)} features")
    ax2.set(title="Features used per particle (selected run)", xlabel="Iteration", ylabel="Number of features",
            ylim=(0, positions.shape[2] + 1))
    ax2.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "17_feature_selection.png")
