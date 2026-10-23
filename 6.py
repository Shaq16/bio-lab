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
