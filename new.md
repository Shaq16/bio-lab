# Bio-Inspired Algorithms Lab - All Codes

## Program 1: Genetic Algorithm for Function Optimization

```python
import math, random
import matplotlib.pyplot as plt

# Objective: maximize f(x) = x*sin(10*pi*x) + 2.0 on [-1, 2] (many peaks)
def f(x):
    return x * math.sin(10 * math.pi * x) + 2.0

def fitness(x):                               # fitness = function value
    return f(x)                               # (for minimizing use -f(x))

pop_size, generations = 30, 100
crossover_rate, mutation_rate, step = 0.9, 0.1, 0.1
low, high = -1, 2

# INITIALIZATION: random values of x
pop = [random.uniform(low, high) for _ in range(pop_size)]

def selection(pop):                           # tournament of size 2
    a, b = random.sample(pop, 2)
    return a if fitness(a) > fitness(b) else b

def crossover(p1, p2):                        # blend crossover
    if random.random() < crossover_rate:
        r = random.random()
        return r * p1 + (1 - r) * p2
    return p1

def mutation(x):                              # small random change
    if random.random() < mutation_rate:
        x += random.uniform(-step, step)
    return min(max(x, low), high)             # keep inside the bounds

best = max(pop, key=fitness)
history, avg_history = [], []
for g in range(generations):
    new_pop = []
    for _ in range(pop_size):
        p1, p2 = selection(pop), selection(pop)
        child = mutation(crossover(p1, p2))
        new_pop.append(child)
    pop = new_pop
    gen_best = max(pop, key=fitness)
    if fitness(gen_best) > fitness(best):
        best = gen_best
    history.append(fitness(best))
    avg_history.append(sum(fitness(x) for x in pop) / pop_size)
    if g % 10 == 0:
        print("Gen", g, " best x =", round(best, 5), " f(x) =", round(fitness(best), 5),
              " avg f =", round(avg_history[-1], 4))

print("Final best x =", best, " f(x) =", fitness(best))

plt.plot(history, label="Best so far")
plt.plot(avg_history, label="Population average")
plt.legend()
plt.xlabel("Generation"); plt.ylabel("Best fitness")
plt.title("Genetic Algorithm - Function Optimization")
plt.show()
```

## Program 2: Hill Climbing for N-Queens

```python
import random

# state[col] = row of the queen in that column
def fitness(state):                      # number of NON-attacking pairs
    n = len(state)
    attacking = 0
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] == state[j] or abs(state[i] - state[j]) == j - i:
                attacking += 1
    return n * (n - 1) // 2 - attacking

def neighbors(state):                    # move one queen within its column
    result = []
    for col in range(len(state)):
        for row in range(len(state)):
            if row != state[col]:
                nb = state[:]
                nb[col] = row
                result.append(nb)
    return result

def hill_climbing(n):                    # steepest ascent
    current = [random.randint(0, n - 1) for _ in range(n)]
    while True:
        best = max(neighbors(current), key=fitness)
        if fitness(best) <= fitness(current):
            return current               # local maximum
        current = best

def random_restart(n, restarts=500):
    goal = n * (n - 1) // 2
    best = None
    for r in range(restarts):
        sol = hill_climbing(n)
        if best is None or fitness(sol) > fitness(best):
            best = sol
        if fitness(best) == goal:
            print("Solved after", r + 1, "restart(s)")
            break
    return best

def print_board(state):
    for row in range(len(state)):
        print(" ".join("Q" if state[c] == row else "." for c in range(len(state))))

N = 8
solution = random_restart(N)
print("State  :", solution)
print("Fitness:", fitness(solution), "/", N * (N - 1) // 2)
print_board(solution)
```

## Program 3: Simulated Annealing for TSP

```python
import math, random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
cities = np.random.rand(20, 2) * 100          # 20 random cities
n = len(cities)

def dist(a, b):
    return math.dist(cities[a], cities[b])

def tour_cost(tour):                          # path cost (closed loop)
    return sum(dist(tour[i], tour[(i + 1) % n]) for i in range(n))

def neighbor(tour):                           # 2-opt: reverse a segment
    a, b = sorted(random.sample(range(n), 2))
    return tour[:a] + tour[a:b + 1][::-1] + tour[b + 1:]

T, T_min, alpha = 1000.0, 0.001, 0.995        # temperature schedule
current = list(range(n))
random.shuffle(current)
best, history = current[:], []

while T > T_min:
    cand = neighbor(current)
    delta = tour_cost(cand) - tour_cost(current)
    # acceptance probability: always if better, else exp(-delta/T)
    if delta < 0 or random.random() < math.exp(-delta / T):
        current = cand
    if tour_cost(current) < tour_cost(best):
        best = current[:]
    history.append(tour_cost(best))
    T = T * alpha                             # cooling

print("Best tour :", best)
print("Best cost :", round(tour_cost(best), 2))

plt.plot(history)
plt.xlabel("Iteration"); plt.ylabel("Best tour cost")
plt.title("Simulated Annealing - TSP")
plt.show()
```

## Program 4: Ant Colony Optimization for TSP

```python
import math, random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
n = 10
cities = np.random.rand(n, 2) * 100
d = np.array([[math.dist(a, b) for b in cities] for a in cities])   # distances

alpha, beta = 1.0, 2.0        # influence of pheromone, influence of distance
rho = 0.5                     # evaporation rate
Q = 100                       # pheromone deposit constant
n_ants, iters = 10, 100

tau = np.ones((n, n))         # pheromone on every edge

def tour_length(tour):
    return sum(d[tour[i], tour[(i + 1) % n]] for i in range(n))

def build_tour():                             # PATH SELECTION
    start = random.randrange(n)
    tour, visited = [start], {start}
    while len(tour) < n:
        i = tour[-1]
        cand = [j for j in range(n) if j not in visited]
        # probability ~ pheromone^alpha * (1/distance)^beta
        w = [(tau[i, j] ** alpha) * ((1 / d[i, j]) ** beta) for j in cand]
        nxt = random.choices(cand, weights=w)[0]
        tour.append(nxt)
        visited.add(nxt)
    return tour

best_tour, best_len, history = None, float("inf"), []
for it in range(iters):
    tours = [build_tour() for _ in range(n_ants)]

    tau *= (1 - rho)                          # EVAPORATION

    for tour in tours:                        # PHEROMONE UPDATE (deposit)
        L = tour_length(tour)
        if L < best_len:
            best_tour, best_len = tour[:], L
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            tau[a, b] += Q / L                # shorter tour -> more pheromone
            tau[b, a] += Q / L
    history.append(best_len)

print("Best tour  :", best_tour)
print("Best length:", round(best_len, 2))

plt.plot(history)
plt.xlabel("Iteration"); plt.ylabel("Best tour length")
plt.title("Ant Colony Optimization - TSP")
plt.show()
```

## Program 5: Particle Swarm Optimization (Global Minima)

```python
import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x):                             # global minimum 0 at x = 0
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

n, dim, iters = 30, 2, 100
w, c1, c2 = 0.7, 1.5, 1.5                     # inertia, cognitive, social
low, high = -5.12, 5.12

pos = np.random.uniform(low, high, (n, dim))
vel = np.zeros((n, dim))
pbest = pos.copy()                            # each particle's best position
pbest_val = np.array([rastrigin(p) for p in pos])
gbest = pbest[np.argmin(pbest_val)].copy()    # swarm's best position
history = []

for t in range(iters):
    r1 = np.random.rand(n, dim)
    r2 = np.random.rand(n, dim)
    vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)  # velocity
    pos = np.clip(pos + vel, low, high)                                # position
    for i in range(n):
        val = rastrigin(pos[i])
        if val < pbest_val[i]:                # update pBest
            pbest_val[i], pbest[i] = val, pos[i].copy()
    gbest = pbest[np.argmin(pbest_val)].copy()  # update gBest
    history.append(rastrigin(gbest))

print("Best position:", gbest)
print("Best value   :", rastrigin(gbest))

plt.plot(history)
plt.xlabel("Iteration"); plt.ylabel("gBest value")
plt.title("PSO - Rastrigin")
plt.show()
```

## Program 6: Differential Evolution

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def de(func, dim=5, n=30, F=0.8, CR=0.9, gens=200, strategy="rand/1"):
    low, high = -5.12, 5.12
    pop = np.random.uniform(low, high, (n, dim))
    cost = np.array([func(p) for p in pop])
    history = []
    for g in range(gens):
        best = pop[np.argmin(cost)]
        for i in range(n):
            others = [k for k in range(n) if k != i]
            a, b, c = pop[np.random.choice(others, 3, replace=False)]

            # 1. MUTATION (selection of strategy)
            if strategy == "rand/1":
                mutant = a + F * (b - c)
            elif strategy == "best/1":
                mutant = best + F * (b - c)
            else:                              # current-to-best/1
                mutant = pop[i] + F * (best - pop[i]) + F * (b - c)
            mutant = np.clip(mutant, low, high)

            # 2. CROSSOVER (binomial)
            mask = np.random.rand(dim) < CR
            mask[np.random.randint(dim)] = True   # at least one gene from mutant
            trial = np.where(mask, mutant, pop[i])

            # 3. SELECTION (greedy)
            if func(trial) < cost[i]:
                pop[i], cost[i] = trial, func(trial)
        history.append(cost.min())
    return pop[np.argmin(cost)], cost.min(), history

for func in (sphere, rastrigin):
    plt.figure()
    for s in ("rand/1", "best/1", "current-to-best/1"):
        x, f, hist = de(func, strategy=s)
        print(func.__name__, s, "-> best =", f)
        plt.plot(hist, label=s)
    plt.yscale("log"); plt.legend()
    plt.xlabel("Generation"); plt.ylabel("Best cost")
    plt.title("DE on " + func.__name__)
plt.show()
```

## Program 7: Tabu Search for Job Scheduling

```python
import random
import matplotlib.pyplot as plt

job_times = [3, 5, 2, 7, 4, 6, 1, 8]          # processing time of each job

def total_completion_time(seq):               # objective to minimize
    t, total = 0, 0
    for job in seq:
        t += job_times[job]                   # completion time of this job
        total += t
    return total

def neighborhood(seq):                        # NEIGHBORHOOD: all pair swaps
    result = []
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            nb = seq[:]
            nb[i], nb[j] = nb[j], nb[i]
            move = (seq[i], seq[j])           # the two jobs that were swapped
            result.append((nb, move))
    return result

tabu_size, max_iter = 5, 50
current = list(range(len(job_times)))
random.shuffle(current)
best, best_cost = current[:], total_completion_time(current)
tabu_list = []                                # recently used moves
history = []

for it in range(max_iter):
    chosen, chosen_move, chosen_cost = None, None, float("inf")
    for nb, move in neighborhood(current):
        cost = total_completion_time(nb)
        is_tabu = move in tabu_list or move[::-1] in tabu_list
        aspiration = cost < best_cost         # ASPIRATION: tabu allowed if new best
        if (not is_tabu or aspiration) and cost < chosen_cost:
            chosen, chosen_move, chosen_cost = nb, move, cost
    if chosen is None:                        # every move is tabu
        break
    current = chosen
    tabu_list.append(chosen_move)             # TABU LIST update
    if len(tabu_list) > tabu_size:
        tabu_list.pop(0)                      # forget the oldest move
    if chosen_cost < best_cost:
        best, best_cost = current[:], chosen_cost
    history.append(best_cost)

print("Best job order :", best)
print("Total completion time:", best_cost)
print("Check (shortest job first):",
      total_completion_time(sorted(range(len(job_times)), key=lambda j: job_times[j])))

plt.plot(history)
plt.xlabel("Iteration"); plt.ylabel("Best total completion time")
plt.title("Tabu Search - Job Scheduling")
plt.show()
```

## Program 8: Biogeography-Based Optimization (BBO)

```python
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
```

## Program 9: Firefly Algorithm

```python
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
```