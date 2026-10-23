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
