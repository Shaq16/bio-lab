# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 09:50:34 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Benchmark function (Sphere) ---
def sphere(x):
    return np.sum(x ** 2)

# --- Biogeography-Based Optimization with Convergence Tracking ---
def BBO(func, dim=5, pop_size=20, generations=100, bounds=(-5, 5),
        elite_count=2, mutation_rate=0.05):
    
    lower, upper = bounds
    pop = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])

    I, E = 1.0, 1.0  # Max immigration and emigration rates
    best_curve = []  # Track best fitness

    for gen in range(generations):
        # Sort by fitness
        idx = np.argsort(fitness)
        pop, fitness = pop[idx], fitness[idx]

        # Normalize fitness safely
        fit_range = fitness.max() - fitness.min()
        if fit_range == 0:
            norm_fit = np.ones_like(fitness)
        else:
            norm_fit = (fitness.max() - fitness) / (fit_range + 1e-9)

        # Immigration (λ) and Emigration (μ)
        lambda_ = I * (1 - norm_fit)
        mu = E * norm_fit

        new_pop = np.copy(pop)

        # --- Migration ---
        for i in range(pop_size):
            for j in range(dim):
                if np.random.rand() < lambda_[i]:
                    # If all mu = 0, choose donor uniformly
                    if mu.sum() == 0:
                        donor = np.random.randint(0, pop_size)
                    else:
                        probs = mu / mu.sum()
                        donor = np.random.choice(pop_size, p=probs)
                    new_pop[i, j] = pop[donor, j]

        # --- Mutation ---
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                new_pop[i] = np.random.uniform(lower, upper, dim)

        # Evaluate new fitness
        new_fitness = np.array([func(ind) for ind in new_pop])

        # --- Elitism ---
        combined = np.vstack((pop, new_pop))
        combined_fitness = np.concatenate((fitness, new_fitness))
        elite_idx = np.argsort(combined_fitness)[:pop_size]
        pop, fitness = combined[elite_idx], combined_fitness[elite_idx]

        # Track best fitness
        best_curve.append(fitness[0])

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], best_curve

# --- Run BBO ---
best_solution, best_value, best_curve = BBO(sphere, dim=5, generations=200)

# --- Plot Convergence ---
plt.figure(figsize=(8, 5))
plt.plot(best_curve, linewidth=2)
plt.title("BBO Convergence Curve (Sphere Function)")
plt.xlabel("Generations")
plt.ylabel("Best Fitness Value")
plt.grid(True)
plt.show()

print("Best Solution:", best_solution)
print("Best Fitness:", best_value)
