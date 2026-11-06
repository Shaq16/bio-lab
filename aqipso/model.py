"""Random Forest construction and the walk-forward fitness function."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from . import config as cfg

# sklearn defaults, stated explicitly: this is the unoptimised baseline.
BASELINE_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "max_samples": None,
}


def make_rf(params: dict, seed: int = cfg.RF_SEED, n_jobs: int = 1) -> RandomForestRegressor:
    return RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)


def decode(position, space=cfg.SEARCH_SPACE) -> dict:
    """Map a point in the unit hypercube to concrete RF hyperparameters."""
    params = {}
    for u, p in zip(position, space):
        value = p.low + float(u) * (p.high - p.low)
        params[p.name] = int(round(value)) if p.integer else round(value, 3)
    return params


def decode_solution(position, feature_names, space=cfg.SEARCH_SPACE):
    """Split a particle into RF hyperparameters and a feature subset.

    The first len(space) coordinates are hyperparameters. Any further coordinates
    are feature switches (threshold transfer: on when >= FEATURE_THRESHOLD).
    A hyperparameter-only particle keeps every feature.
    """
    position = np.asarray(position, float)
    n_hp = len(space)
    params = decode(position[:n_hp], space)
    names = list(feature_names)
    if len(position) == n_hp:
        return params, names
    switches = position[n_hp:]
    keep = switches >= cfg.FEATURE_THRESHOLD
    if keep.sum() < cfg.MIN_FEATURES:
        keep = np.zeros(len(switches), bool)
        keep[np.argsort(-switches)[:cfg.MIN_FEATURES]] = True
    return params, [f for f, k in zip(names, keep) if k]


def cv_rmse(params: dict, X, y, n_splits: int = cfg.CV_FOLDS, n_jobs: int = 1) -> float:
    """Mean RMSE over expanding-window folds: always train on past, score on future."""
    scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = make_rf(params, n_jobs=n_jobs)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(root_mean_squared_error(y.iloc[val_idx], model.predict(X.iloc[val_idx])))
    return float(np.mean(scores))


class Fitness:
    """PSO objective: decodes a position and memoises repeated solutions.

    Different continuous positions can round to identical hyperparameters and
    feature subsets, so the cache avoids refitting them. `calls` still counts every
    evaluation requested, which keeps the budget comparable across optimisers.
    """

    def __init__(self, X, y):
        self.X, self.y = X, y
        self.cache = {}
        self.records = []        # one row per unique solution, for later analysis
        self.calls = 0

    def __call__(self, position) -> float:
        self.calls += 1
        params, features = decode_solution(position, self.X.columns)
        key = (tuple(sorted(params.items())), tuple(features))
        if key not in self.cache:
            score = cv_rmse(params, self.X[features], self.y)
            self.cache[key] = score
            self.records.append(params | {"n_features": len(features), "cv_rmse": score})
        return self.cache[key]
