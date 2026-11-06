"""Run the full experiment and write every table, metric and figure in results/.

    python run_experiment.py                full run
    python run_experiment.py --quick        small budget, written to results/quick/
    python run_experiment.py --skip-search  reuse saved search runs, redo the rest
    python run_experiment.py --reuse-hp-only  reuse saved hyperparameter-only runs

Four searches are run for every seed. "PSO" and "Random search" tune the six
hyperparameters and select features at the same time; the two
"(hyperparameters only)" searches keep all features and act as an ablation.
"""
import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error

from aqipso import config as cfg
from aqipso import analysis, plots
from aqipso.data import build_dataset
from aqipso.evaluate import all_metrics, block_bootstrap_rmse_gain, regression_metrics
from aqipso.model import BASELINE_PARAMS, Fitness, cv_rmse, decode_solution, make_rf
from aqipso.pso import ParticleSwarm, random_search


HP_ONLY = " (hyperparameters only)"
METHODS = ("PSO", "Random search", "PSO" + HP_ONLY, "Random search" + HP_ONLY)


def run_search(method, seed, X, y, pso_cfg):
    """One independent search. Each sees training data only."""
    start = time.time()
    fitness = Fitness(X, y)
    dim = len(cfg.SEARCH_SPACE) + (0 if method.endswith(HP_ONLY) else X.shape[1])
    if method.startswith("PSO"):
        seen = []

        def traced(position):
            score = fitness(position)
            seen.append(score)
            return score

        res = ParticleSwarm(traced, dim, pso_cfg, seed).run()
        out = {
            "position": res.best_position.tolist(), "cv_rmse": res.best_fitness,
            "gbest_history": res.gbest_history, "mean_history": res.mean_history,
            "diversity_history": res.diversity_history, "inertia_history": res.inertia_history,
            "positions_history": [p.tolist() for p in res.positions_history],
            "best_by_evaluation": np.minimum.accumulate(seen).tolist(),
        }
    else:
        budget = pso_cfg.n_particles * (pso_cfg.n_iterations + 1)
        position, score, curve = random_search(fitness, dim, budget, seed)
        out = {"position": position.tolist(), "cv_rmse": score, "best_by_evaluation": curve}
    out["all_evaluations"] = fitness.records
    params, features = decode_solution(out["position"], X.columns)
    out.update(method=method, seed=seed, params=params, features=features, evaluations=fitness.calls,
               unique_fits=len(fitness.cache), seconds=round(time.time() - start, 1))
    print(f"  {method} seed {seed}: CV RMSE {out['cv_rmse']:.3f} in {out['seconds']}s", flush=True)
    return out


def fit(params, features, ds, seed=cfg.RF_SEED):
    return make_rf(params, seed=seed, n_jobs=-1).fit(ds.X_train[features], ds.y_train)


def test_scores(params, features, ds, seeds=cfg.EVAL_SEEDS):
    """Refit on all training data under several forest seeds, then score on test."""
    scores = []
    for s in seeds:
        model = fit(params, features, ds, seed=s)
        scores.append(regression_metrics(ds.y_test, model.predict(ds.X_test[features])))
    return pd.DataFrame(scores)


def curve(params, features, ds, fractions=(0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)):
    """Learning curve that respects time.

    The validation window stays fixed at the last 20% of the training period, and
    each model trains on the most recent days before that window.
    """
    split = int(len(ds.X_train) * 0.8)
    X_val, y_val = ds.X_train[features].iloc[split:], ds.y_train.iloc[split:]
    sizes, train, val = [], [], []
    for fraction in fractions:
        start = split - int(split * fraction)
        X_fit, y_fit = ds.X_train[features].iloc[start:split], ds.y_train.iloc[start:split]
        model = make_rf(params, n_jobs=-1).fit(X_fit, y_fit)
        sizes.append(len(X_fit))
        train.append(root_mean_squared_error(y_fit, model.predict(X_fit)))
        val.append(root_mean_squared_error(y_val, model.predict(X_val)))
    return {"sizes": sizes, "train": train, "val": val}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--reuse-hp-only", action="store_true")
    args = ap.parse_args()

    pso_cfg = cfg.PSOConfig()
    seeds = cfg.SEARCH_SEEDS
    if args.quick:
        # Quick runs get their own folder so they can never replace the real results.
        pso_cfg, seeds = replace(pso_cfg, n_particles=4, n_iterations=3), seeds[:2]
        cfg.RESULTS_DIR = cfg.RESULTS_DIR / "quick"
        cfg.FIGURES_DIR = cfg.RESULTS_DIR / "figures"
        cfg.TABLES_DIR = cfg.RESULTS_DIR / "tables"
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    ds = build_dataset()
    print("Data:", json.dumps({k: ds.report[k] for k in ("city", "train_samples", "test_samples", "n_features")}))

    # 1. Unoptimised baselines.
    baseline_cv = cv_rmse(BASELINE_PARAMS, ds.X_train, ds.y_train, n_jobs=-1)
    print(f"Default RF walk-forward CV RMSE: {baseline_cv:.3f}")

    # 2. Search. Runs are parallel; forests inside each run stay single-threaded.
    runs_path = cfg.RESULTS_DIR / "search_runs.json"
    if args.skip_search:
        runs = json.loads(runs_path.read_text())
    else:
        methods = METHODS
        kept = []
        if args.reuse_hp_only:
            kept = [r for r in json.loads(runs_path.read_text()) if r["method"].endswith(HP_ONLY)]
            methods = [m for m in METHODS if not m.endswith(HP_ONLY)]
        jobs = [(m, s) for m in methods for s in seeds]
        runs = Parallel(n_jobs=min(len(jobs), 14))(
            delayed(run_search)(m, s, ds.X_train, ds.y_train, pso_cfg) for m, s in jobs) + kept
        runs_path.write_text(json.dumps(runs))
    all_features = list(ds.X_train.columns)

    # 3. Selection uses validation error only; the test set plays no part in it.
    # The final model was fixed in advance as the best joint PSO run.
    best_of = lambda m: min((r for r in runs if r["method"] == m), key=lambda r: r["cv_rmse"])
    best_run, best_hp_run = best_of("PSO"), best_of("PSO" + HP_ONLY)
    tuned_params, tuned_features = best_run["params"], best_run["features"]
    print("Selected PSO hyperparameters:", tuned_params)
    print(f"Selected {len(tuned_features)} of {len(all_features)} features:", tuned_features)

    # 4. Final models and predictions on the untouched test period.
    baseline = fit(BASELINE_PARAMS, all_features, ds)
    tuned = fit(tuned_params, tuned_features, ds)
    tuned_hp = fit(best_hp_run["params"], all_features, ds)
    pred = pd.DataFrame({
        "actual": ds.y_test, "persistence": ds.X_test["AQI"],
        "baseline": baseline.predict(ds.X_test), "tuned_hp": tuned_hp.predict(ds.X_test),
        "tuned": tuned.predict(ds.X_test[tuned_features])})
    pred.to_csv(cfg.TABLES_DIR / "test_predictions.csv")

    names = {"Persistence": "persistence", "Default RF": "baseline",
             "PSO RF (hyperparameters only)": "tuned_hp", "PSO-tuned RF": "tuned"}
    comparison = pd.DataFrame({n: all_metrics(pred["actual"], pred[c]) for n, c in names.items()}).T
    comparison.round(4).to_csv(cfg.TABLES_DIR / "baseline_vs_tuned.csv")

    # 5. Robustness: forest-seed noise, every search run, and sub-periods.
    base_scores = test_scores(BASELINE_PARAMS, all_features, ds)
    rows = [{"method": "Default RF", "seed": "-", "cv_rmse": baseline_cv, **base_scores.mean().round(3).to_dict(),
             "RMSE_sd": base_scores["RMSE"].std(), "n_features": len(all_features)}]
    for r in runs:
        s = test_scores(r["params"], r["features"], ds)
        rows.append({"method": r["method"], "seed": r["seed"], "cv_rmse": r["cv_rmse"], **s.mean().round(3).to_dict(),
                     "RMSE_sd": s["RMSE"].std(), "n_features": len(r["features"]), **r["params"]})
    robustness = pd.DataFrame(rows)
    robustness.round(3).to_csv(cfg.TABLES_DIR / "search_runs_summary.csv", index=False)

    winter = pred.index.month.isin([10, 11, 12, 1, 2])
    periods = {"Whole test": pred, "2024": pred.loc["2024"], "2025": pred.loc["2025"], "2026 (to 24 Oct)": pred.loc["2026"],
               "Winter (Oct to Feb)": pred[winter], "Rest of year (Mar to Sep)": pred[~winter]}
    by_period = pd.DataFrame([
        {"period": n, "days": len(f), "mean_aqi": f["actual"].mean(), "persistence_RMSE": regression_metrics(f["actual"], f["persistence"])["RMSE"], **{f"{c}_{k}": regression_metrics(f["actual"], f[c])[k] for c in ("baseline", "tuned_hp", "tuned") for k in ("RMSE", "MAE")}}
        for n, f in periods.items()])
    by_period.round(3).to_csv(cfg.TABLES_DIR / "error_by_period.csv", index=False)

    # Traditional models, untuned, for context.
    extra = analysis.traditional_models(ds)
    for name, p in extra.items():
        pred[name] = p
    full = pd.DataFrame({n: regression_metrics(pred["actual"], pred[c]) for n, c in
                         {**names, **{k: k for k in extra}}.items()}).T
    full.round(4).to_csv(cfg.TABLES_DIR / "all_models.csv")
    pred.to_csv(cfg.TABLES_DIR / "test_predictions.csv")

    stats = analysis.residual_stats(pred["actual"] - pred["tuned"])
    signif = analysis.significance(pred["actual"], pred["baseline"], pred["tuned"])
    hp_corr = analysis.hyperparameter_correlation(runs)
    hp_corr.round(3).to_csv(cfg.TABLES_DIR / "hyperparameter_fitness_correlation.csv", header=["spearman"])
    bench = analysis.benchmarks()
    sens = analysis.sensitivity()
    sens.round(6).to_csv(cfg.TABLES_DIR / "pso_sensitivity.csv", index=False)

    importance = pd.Series(tuned.feature_importances_, index=tuned_features).sort_values(ascending=False)
    importance.round(4).to_csv(cfg.TABLES_DIR / "feature_importance.csv", header=["importance"])

    pso_joint = [r for r in runs if r["method"] == "PSO"]
    feature_freq = pd.Series({f: np.mean([f in r["features"] for r in pso_joint]) for f in all_features})
    feature_freq.round(3).to_csv(cfg.TABLES_DIR / "feature_selection_frequency.csv", header=["share_of_pso_runs"])

    summary = lambda m: {"cv_rmse_mean": robustness.query("method == @m")["cv_rmse"].mean(),
                         "cv_rmse_best": robustness.query("method == @m")["cv_rmse"].min(),
                         "test_rmse_mean": robustness.query("method == @m")["RMSE"].mean(),
                         "test_rmse_sd_between_runs": robustness.query("method == @m")["RMSE"].std()}
    metrics = {
        "data": ds.report, "pso_config": asdict(pso_cfg),
        "baseline_params": BASELINE_PARAMS, "tuned_params": tuned_params,
        "tuned_features": tuned_features, "dropped_features": [f for f in all_features if f not in tuned_features],
        "hp_only_params": best_hp_run["params"], "hp_only_cv_rmse": best_hp_run["cv_rmse"],
        "baseline_cv_rmse": baseline_cv, "tuned_cv_rmse": best_run["cv_rmse"],
        "cv_improvement_pct": (baseline_cv - best_run["cv_rmse"]) / baseline_cv * 100,
        "test_comparison": comparison.to_dict("index"),
        "rmse_improvement_pct": float((comparison.loc["Default RF", "RMSE"] - comparison.loc["PSO-tuned RF", "RMSE"]) / comparison.loc["Default RF", "RMSE"] * 100),
        "default_rf_test_rmse_over_10_forest_seeds": {"mean": base_scores["RMSE"].mean(), "sd": base_scores["RMSE"].std()},
        "pso_summary": summary("PSO"), "random_search_summary": summary("Random search"),
        "pso_hp_only_summary": summary("PSO" + HP_ONLY), "random_search_hp_only_summary": summary("Random search" + HP_ONLY),
        "feature_selection_frequency": feature_freq.to_dict(),
        "bootstrap_tuned_vs_hp_only": block_bootstrap_rmse_gain(pred["actual"], pred["tuned_hp"], pred["tuned"]),
        "significance_tuned_vs_hp_only": analysis.significance(pred["actual"], pred["tuned_hp"], pred["tuned"]),
        "bootstrap_tuned_vs_default": block_bootstrap_rmse_gain(pred["actual"], pred["baseline"], pred["tuned"]),
        "residuals": stats, "significance": signif,
        "benchmarks": {k: {kk: vv for kk, vv in v.items() if kk != "curves"} for k, v in bench.items()},
        "runtime_seconds": {m: [r["seconds"] for r in runs if r["method"] == m] for m in METHODS},
        "bootstrap_tuned_vs_persistence": block_bootstrap_rmse_gain(pred["actual"], pred["persistence"], pred["tuned"]),
    }
    (cfg.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # 6. Figures.
    curves = {"baseline": curve(BASELINE_PARAMS, all_features, ds), "tuned": curve(tuned_params, tuned_features, ds)}
    plots.data_overview(ds.raw, ds)
    plots.missing_values(ds.report)
    plots.correlations(ds)
    plots.convergence(runs, baseline_cv)
    plots.pso_vs_random(runs, baseline_cv)
    plots.swarm_dynamics(best_run)
    plots.parameter_paths(best_run)
    plots.metric_comparison(comparison)
    plots.forecast_timeline(pred)
    plots.actual_vs_predicted(pred)
    plots.residuals(pred)
    plots.learning_curves(curves)
    plots.feature_importance(importance)
    plots.confusion(pred)
    plots.model_comparison(full)
    plots.benchmark_convergence(bench)
    plots.feature_selection(feature_freq, tuned_features, best_run)

    print(comparison.round(3).T.to_string())
    print(json.dumps({k: metrics[k] for k in ("cv_improvement_pct", "rmse_improvement_pct", "bootstrap_tuned_vs_default", "bootstrap_tuned_vs_hp_only", "pso_summary", "random_search_summary", "pso_hp_only_summary")}, indent=1))


if __name__ == "__main__":
    main()
