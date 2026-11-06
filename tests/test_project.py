"""Checks on the PSO operators and on the absence of data leakage."""
import numpy as np
import pandas as pd
import pytest

from aqipso.config import PSOConfig, SEARCH_SPACE
from aqipso.data import build_dataset, clean, engineer_features, load_raw
from aqipso.evaluate import to_band
from aqipso.model import decode
from aqipso.pso import ParticleSwarm


def sphere(x):
    return float(np.sum((np.asarray(x) - 0.7) ** 2))


def test_pso_finds_known_optimum():
    result = ParticleSwarm(sphere, dim=6, config=PSOConfig(n_particles=20, n_iterations=60), seed=1).run()
    assert result.best_fitness < 1e-4
    assert np.allclose(result.best_position, 0.7, atol=0.02)


def test_pso_gbest_never_worsens():
    history = ParticleSwarm(sphere, dim=6, seed=2).run().gbest_history
    assert all(b <= a for a, b in zip(history, history[1:]))


def test_pso_is_reproducible():
    a = ParticleSwarm(sphere, dim=4, seed=3).run()
    b = ParticleSwarm(sphere, dim=4, seed=3).run()
    assert np.array_equal(a.best_position, b.best_position)


def test_particles_stay_in_bounds():
    x = np.array([[0.95, 0.05, 0.5]])
    v = np.array([[0.2, -0.2, 0.0]])
    moved, velocity = ParticleSwarm.update_position(x, v)
    assert np.allclose(moved, [[0.85, 0.15, 0.5]])
    assert np.allclose(velocity, [[-0.2, 0.2, 0.0]])


def test_velocity_is_clamped():
    swarm = ParticleSwarm(sphere, dim=2, config=PSOConfig(v_max=0.1))
    x = np.zeros((1, 2))
    v = swarm.update_velocity(np.ones((1, 2)) * 5, x, np.ones((1, 2)), np.ones((1, 2)), 0.9)
    assert np.all(np.abs(v) <= 0.1)


def test_inertia_decreases_linearly():
    swarm = ParticleSwarm(sphere, dim=2, config=PSOConfig(n_iterations=11))
    assert swarm.inertia(0) == pytest.approx(0.9)
    assert swarm.inertia(10) == pytest.approx(0.4)
    assert swarm.inertia(5) == pytest.approx(0.65)


@pytest.mark.parametrize("u", [0.0, 0.5, 1.0])
def test_decode_respects_search_space(u):
    params = decode([u] * len(SEARCH_SPACE))
    for p in SEARCH_SPACE:
        assert p.low <= params[p.name] <= p.high
        assert isinstance(params[p.name], int) == p.integer


def test_split_is_chronological_and_target_is_next_day():
    ds = build_dataset()
    assert ds.X_train.index.max() < ds.X_test.index.min()
    day = ds.X_test.index[0]
    assert ds.y_test.loc[day] == ds.raw.loc[day + pd.Timedelta(days=1), "AQI"]


def test_target_is_never_imputed():
    ds = build_dataset()
    assert ds.raw["AQI"].isna().sum() > 0
    observed = ds.raw["AQI"].dropna()
    next_days = ds.y_train.index + pd.Timedelta(days=1)
    assert next_days.isin(observed.index).all()


def test_features_do_not_see_the_future():
    raw = load_raw()
    cut = pd.Timestamp("2018-06-15")
    full = engineer_features(clean(raw)[0])
    changed = raw.copy()
    changed.loc[changed.index > cut, raw.select_dtypes("number").columns] = 9999.0
    altered = engineer_features(clean(changed)[0])
    pd.testing.assert_series_equal(full.loc[cut], altered.loc[cut])


def test_aqi_band_edges():
    assert to_band([50, 51, 100, 101, 200, 300, 400, 401]).tolist() == [
        "Good", "Satisfactory", "Satisfactory", "Moderate", "Moderate", "Poor",
        "Very Poor", "Severe"]


def test_feature_switches_use_threshold():
    from aqipso.model import decode_solution
    names = [f"f{i}" for i in range(5)]
    position = [0.5] * len(SEARCH_SPACE) + [0.9, 0.2, 0.5, 0.49, 0.7]
    params, features = decode_solution(position, names)
    assert features == ["f0", "f2", "f4"]
    assert params == decode([0.5] * len(SEARCH_SPACE))


def test_feature_selection_keeps_a_minimum():
    from aqipso.config import MIN_FEATURES
    from aqipso.model import decode_solution
    names = [f"f{i}" for i in range(6)]
    position = [0.5] * len(SEARCH_SPACE) + [0.1, 0.4, 0.0, 0.3, 0.2, 0.05]
    _, features = decode_solution(position, names)
    assert len(features) == MIN_FEATURES
    assert set(features) == {"f1", "f3", "f4"}


def test_hyperparameter_only_particle_keeps_all_features():
    from aqipso.model import decode_solution
    _, features = decode_solution([0.3] * len(SEARCH_SPACE), ["a", "b"])
    assert features == ["a", "b"]
