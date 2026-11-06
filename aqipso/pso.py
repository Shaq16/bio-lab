"""Particle Swarm Optimisation, implemented from first principles.

Particles live in the unit hypercube [0, 1]^d, so dimensions with very different
scales (tree count vs. feature fraction) move at comparable speed. Each iteration:

    w_k     = w_start - (w_start - w_end) * k / (K - 1)                    (inertia)
    v[k+1]  = w_k v[k] + c1 r1 (pbest - x[k]) + c2 r2 (gbest - x[k])       (velocity)
    v[k+1]  = clip(v[k+1], -v_max, +v_max)                                 (clamping)
    x[k+1]  = x[k] + v[k+1]                                                (position)

Then particles leaving the box are reflected, fitness is evaluated, and the personal
and global bests are updated.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import PSOConfig


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    gbest_history: list = field(default_factory=list)  # best fitness after each iteration
    mean_history: list = field(default_factory=list)   # swarm mean fitness per iteration
    diversity_history: list = field(default_factory=list)  # mean distance to swarm centre
    inertia_history: list = field(default_factory=list)
    positions_history: list = field(default_factory=list)  # position array per iteration
    evaluations: int = 0


class ParticleSwarm:
    """Global-best PSO minimising `objective` over the unit hypercube."""

    def __init__(self, objective: Callable, dim: int, config: PSOConfig = PSOConfig(), seed: int = 0):
        self.objective = objective
        self.dim = dim
        self.cfg = config
        self.rng = np.random.default_rng(seed)

    def inertia(self, iteration: int) -> float:
        """Linearly decreasing inertia: explore early, exploit late."""
        span = max(self.cfg.n_iterations - 1, 1)
        return self.cfg.w_start - (self.cfg.w_start - self.cfg.w_end) * iteration / span

    def update_velocity(self, v, x, pbest, gbest, w):
        """Combine inertia, cognitive and social terms, then clamp the speed."""
        r1 = self.rng.random(x.shape)
        r2 = self.rng.random(x.shape)
        cognitive = self.cfg.c1 * r1 * (pbest - x)
        social = self.cfg.c2 * r2 * (gbest - x)
        return np.clip(w * v + cognitive + social, -self.cfg.v_max, self.cfg.v_max)

    @staticmethod
    def update_position(x, v):
        """Move particles, reflecting off the walls and reversing their velocity."""
        x_new = x + v
        for _ in range(2):  # v_max < 1, so two reflections always land inside the box
            below, above = x_new < 0.0, x_new > 1.0
            x_new = np.where(below, -x_new, x_new)
            x_new = np.where(above, 2.0 - x_new, x_new)
            v = np.where(below | above, -v, v)
        return np.clip(x_new, 0.0, 1.0), v

    def _evaluate(self, x) -> np.ndarray:
        return np.array([self.objective(p) for p in x])

    def run(self) -> PSOResult:
        c = self.cfg
        # Initialise: random positions, small random velocities.
        x = self.rng.random((c.n_particles, self.dim))
        v = self.rng.uniform(-c.v_max, c.v_max, x.shape)
        fitness = self._evaluate(x)

        pbest, pbest_fit = x.copy(), fitness.copy()
        best = int(np.argmin(pbest_fit))
        gbest, gbest_fit = pbest[best].copy(), float(pbest_fit[best])

        result = PSOResult(gbest.copy(), gbest_fit)
        result.evaluations = c.n_particles

        for k in range(c.n_iterations):
            w = self.inertia(k)
            v = self.update_velocity(v, x, pbest, gbest, w)
            x, v = self.update_position(x, v)
            fitness = self._evaluate(x)
            result.evaluations += c.n_particles

            # Personal best: each particle remembers its best position so far.
            improved = fitness < pbest_fit
            pbest[improved] = x[improved]
            pbest_fit[improved] = fitness[improved]

            # Global best: shared with the whole swarm on the next iteration.
            best = int(np.argmin(pbest_fit))
            if pbest_fit[best] < gbest_fit:
                gbest, gbest_fit = pbest[best].copy(), float(pbest_fit[best])

            result.gbest_history.append(gbest_fit)
            result.mean_history.append(float(fitness.mean()))
            result.diversity_history.append(
                float(np.linalg.norm(x - x.mean(axis=0), axis=1).mean()))
            result.inertia_history.append(w)
            result.positions_history.append(x.copy())

        result.best_position, result.best_fitness = gbest, gbest_fit
        return result


def random_search(objective: Callable, dim: int, n_evaluations: int, seed: int = 0):
    """Uniform random search with the same budget, used as a control for PSO."""
    rng = np.random.default_rng(seed)
    points = rng.random((n_evaluations, dim))
    scores = np.array([objective(p) for p in points])
    running_best = np.minimum.accumulate(scores)
    best = int(np.argmin(scores))
    return points[best], float(scores[best]), running_best.tolist()
