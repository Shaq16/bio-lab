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
