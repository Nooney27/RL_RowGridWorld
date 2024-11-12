import RowGridWorld
import numpy as np

grid = RowGridWorld.GridWorld(size=5)

def compute_q_from_s_a(env, V, s, a, gamma):
    q = 0
    for prob, next_state, reward, done in env.P[s][a]:
        q += prob * (reward + gamma * V[next_state])
    return q

V = np.zeros([grid.nS, 1])
gamma = 0.99
theta = 0.0001 

i = 0
while True:
    i += 1
    delta = 0

    for s in range(grid.nS):
        q_s = np.zeros([grid.nA, 1])

        for a in range(grid.nA):
            q_s[a] = compute_q_from_s_a(grid, V, s, a, gamma)
        
        new_v = np.max(q_s)
        delta = max(delta, np.abs(V[s] - new_v))
        V[s] = new_v

    if delta < theta:
        print("Converged after", i, "iterations")
        break
print("Optimal Value Function:", V)