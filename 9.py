# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 09:51:55 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Benchmark: Rastrigin (multimodal, global min = 0 at x=0)
# -------------------------
def rastrigin(x):
    x = np.asarray(x)
    A = 10.0
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# -------------------------
# Firefly Algorithm
# -------------------------
def firefly_algorithm(func,
                      dim=10,
                      n_fireflies=40,
                      max_iter=200,
                      lb=-5.12,
                      ub=5.12,
                      alpha=0.2,       # randomness (step size)
                      alpha_decay=0.98,# decay per generation (optional)
                      beta0=1.0,       # attractiveness at r=0
                      gamma=1.0,       # light absorption coefficient
                      seed=None):
    """
    Returns best_solution, best_fitness, history_of_best
    """
    rng = np.random.default_rng(seed)

    # Initialize population uniformly in bounds
    X = rng.uniform(lb, ub, size=(n_fireflies, dim))
    fitness = np.array([func(x) for x in X])

    # For minimization: brightness = 1 / (1 + fitness)
    def brightness(f):
        return 1.0 / (1.0 + f)

    best_history = []
    best_idx = np.argmin(fitness)
    best_sol = X[best_idx].copy()
    best_fit = fitness[best_idx]
    best_history.append(best_fit)

    for t in range(max_iter):
        # Sort fireflies by fitness (ascending -> brighter first)
        idx = np.argsort(fitness)
        X = X[idx]
        fitness = fitness[idx]
        B = brightness(fitness)  # brightness array

        # Movement: each firefly i moves toward firefly j if j is brighter
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] < fitness[i]:
                    # distance between i and j
                    r = np.linalg.norm(X[i] - X[j])
                    beta = beta0 * np.exp(-gamma * (r**2))
                    # Move i toward j
                    step = beta * (X[j] - X[i])
                    random_vec = alpha * (rng.random(dim) - 0.5) * (ub - lb)
                    X[i] = X[i] + step + random_vec

                    # enforce bounds (simple clipping)
                    X[i] = np.clip(X[i], lb, ub)

                    # evaluate new fitness for i (immediate)
                    fitness[i] = func(X[i])

        # update best
        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_fit:
            best_fit = fitness[cur_best_idx]
            best_sol = X[cur_best_idx].copy()

        best_history.append(best_fit)

        # optionally decay alpha
        alpha *= alpha_decay

    return best_sol, best_fit, best_history

# -------------------------
# Run the algorithm (example)
# -------------------------
if __name__ == "__main__":
    np.random.seed(0)
    best_sol, best_fit, history = firefly_algorithm(
        func=rastrigin,
        dim=10,
        n_fireflies=40,
        max_iter=300,
        lb=-5.12,
        ub=5.12,
        alpha=0.5,
        alpha_decay=0.995,
        beta0=1.0,
        gamma=0.5,
        seed=42
    )

    print("Best fitness:", best_fit)
    print("Best solution (first 6 dims):", best_sol[:6])

    # Plot convergence
    plt.figure(figsize=(8,5))
    plt.plot(history, linewidth=2)
    plt.yscale('log')    # Rastrigin often benefits from log-scale viewing
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (log scale)")
    plt.title("Firefly Algorithm Convergence (Rastrigin, dim=10)")
    plt.grid(True)
    plt.show()
