import RowGridWorld
import numpy as np


grid = RowGridWorld.GridWorld(size=5)

pi = np.ones((grid.nS, grid.nA)) * 0.5
V = np.zeros([grid.nS, 1])
gamma = 0.99
theta = 0.0001

def compute_q_from_s_a(env, V, s, a, gamma):
    q = 0
    for prob, next_state, reward, done in env.P[s][a]:
        q += prob * (reward + gamma * V[next_state])
    return q

V_history = []  # To store value function at each iteration
deltas = []     # To store delta at each iteration

i = 0
while True:
    i += 1
    delta = 0
    V_new = np.zeros_like(V)
    for s in range(grid.nS):
        V_s = 0
        for a in range(grid.nA):
            prob_a = pi[s][a]
            q_s_a = compute_q_from_s_a(grid, V, s, a, gamma)
            V_s += prob_a * q_s_a
        V_new[s] = V_s
        delta = max(delta, np.abs(V[s] - V_new[s]))
    V = V_new
    V_history.append(V.copy())
    deltas.append(delta)
    if delta < theta:
        print(f'Converged after {i} iterations')
        break


grid.plot_gridworld()
grid.display_transition_table()
print("Final Value Function:")
print(V)

# Plotting the Value Function after Convergence
# This plot shows the estimated value of each state after the policy evaluation has converged.
# The values represent the expected return starting from that state under the given policy.
RowGridWorld.plot_value_function(grid, V)

# Plotting the Value Function as a Heatmap
# This heatmap provides a visual representation of the value function.
# Higher values are indicated by warmer colors, allowing for quick identification of high-value states.
RowGridWorld.plot_value_heatmap(V)

# Plotting the Convergence of the Value Function over Iterations
# This plot shows how the value of each state changes over each iteration.
# It helps to visualize the convergence process of the value function.
RowGridWorld.plot_value_convergence(V_history)








