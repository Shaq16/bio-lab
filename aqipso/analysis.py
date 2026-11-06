"""Supporting analyses for the report: extra baselines, statistics and PSO benchmarks."""
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as cfg
from .config import PSOConfig
from .pso import ParticleSwarm


def traditional_models(ds) -> dict:
    """Untuned linear, kernel and boosting models, trained on the same features."""
    models = {
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Support vector regression": make_pipeline(StandardScaler(), SVR(C=100, epsilon=5)),
        "Gradient boosting": GradientBoostingRegressor(random_state=cfg.RF_SEED),
    }
    return {name: m.fit(ds.X_train, ds.y_train).predict(ds.X_test) for name, m in models.items()}


def residual_stats(residuals: pd.Series) -> dict:
    r = np.asarray(residuals, float)
    sd = r.std(ddof=1)
    return {
        "mean": float(r.mean()), "sd": float(sd), "skewness": float(st.skew(r)),
        "kurtosis": float(st.kurtosis(r, fisher=False)),
        "shapiro_p": float(st.shapiro(r).pvalue),
        "share_beyond_2sd_pct": float((np.abs(r - r.mean()) > 2 * sd).mean() * 100),
    }


def significance(y, pred_a, pred_b) -> dict:
    """Paired tests on squared errors of model a (baseline) versus model b (tuned)."""
    ea = (np.asarray(y) - np.asarray(pred_a)) ** 2
    eb = (np.asarray(y) - np.asarray(pred_b)) ** 2
    d = ea - eb
    t = st.ttest_rel(ea, eb)
    w = st.wilcoxon(ea, eb)
    return {"mean_sq_error_reduction": float(d.mean()), "t_statistic": float(t.statistic),
            "t_p_value": float(t.pvalue), "wilcoxon_statistic": float(w.statistic),
            "wilcoxon_p_value": float(w.pvalue), "days_tuned_better_pct": float((d > 0).mean() * 100)}


def hyperparameter_correlation(runs) -> pd.Series:
    """Spearman correlation of each hyperparameter with validation RMSE over every PSO evaluation."""
    rows = [e for r in runs if r["method"] == "PSO" for e in r["all_evaluations"]]
    df = pd.DataFrame(rows)
    return df.drop(columns="cv_rmse").corrwith(df["cv_rmse"], method="spearman").sort_values(key=abs, ascending=False)


# Standard optimisation test functions, rescaled so the unit cube maps to each usual domain.
def sphere(u):
    x = (np.asarray(u) - 0.5) * 200          # [-100, 100]
    return float(np.sum(x ** 2))


def rastrigin(u):
    x = (np.asarray(u) - 0.5) * 10.24        # [-5.12, 5.12]
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(u):
    x = (np.asarray(u) - 0.5) * 10           # [-5, 5]
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


BENCHMARKS = {"Sphere": sphere, "Rastrigin": rastrigin, "Rosenbrock": rosenbrock}


def benchmarks(dim: int = 6, seeds=range(5), config: PSOConfig = PSOConfig(n_particles=30, n_iterations=200)) -> dict:
    out = {}
    for name, f in BENCHMARKS.items():
        res = [ParticleSwarm(f, dim, config, s).run() for s in seeds]
        best = np.array([r.best_fitness for r in res])
        out[name] = {"best_mean": float(best.mean()), "best_sd": float(best.std(ddof=1)),
                     "best_min": float(best.min()), "curves": [r.gbest_history for r in res]}
    return out


def sensitivity(dim: int = 6, seeds=range(10)) -> pd.DataFrame:
    """Effect of swarm size, inertia schedule and acceleration constants on Rastrigin."""
    base = PSOConfig(n_particles=20, n_iterations=100)
    variants = {
        "Base (20 particles, w 0.9 to 0.4, c1 = c2 = 1.5)": base,
        "10 particles": PSOConfig(n_particles=10, n_iterations=100),
        "40 particles": PSOConfig(n_particles=40, n_iterations=100),
        "Constant inertia w = 0.7": PSOConfig(n_particles=20, n_iterations=100, w_start=0.7, w_end=0.7),
        "Constant inertia w = 0.4": PSOConfig(n_particles=20, n_iterations=100, w_start=0.4, w_end=0.4),
        "c1 = 2.0, c2 = 1.0 (more cognitive)": PSOConfig(n_particles=20, n_iterations=100, c1=2.0, c2=1.0),
        "c1 = 1.0, c2 = 2.0 (more social)": PSOConfig(n_particles=20, n_iterations=100, c1=1.0, c2=2.0),
    }
    rows = []
    for name, c in variants.items():
        best = np.array([ParticleSwarm(rastrigin, dim, c, s).run().best_fitness for s in seeds])
        rows.append({"variant": name, "mean_best": best.mean(), "sd_best": best.std(ddof=1), "min_best": best.min()})
    return pd.DataFrame(rows)
