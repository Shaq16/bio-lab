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
