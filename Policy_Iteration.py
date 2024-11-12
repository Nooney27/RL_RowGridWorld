import RowGridWorld
import numpy as np

grid = RowGridWorld.GridWorld(size=5)

def compute_q_from_s_a(env, V, s, a, gamma):
    q = 0
    for prob, next_state, reward, done in env.P[s][a]:
        q += prob * (reward + gamma * V[next_state])
    return q

def policy_evaluation(env, pi, V, gamma, theta):
    V_updated = np.copy(V)
    improvement = True

    while True:
        delta = 0
        for s in range(env.nS):
            v_new = 0

            for a in range(env.nA):
                prob_a = pi[s][a]
                q_s_a = compute_q_from_s_a(env, V_updated, s, a, gamma)
                v_new += prob_a * q_s_a
            
            delta = max(delta, np.abs(V_updated[s] - v_new))
            V_updated[s] = v_new
        
        if delta < theta:
            break
    if np.array_equal(V, V_updated):
        improvement = False
    return V_updated, improvement

def improve_policy(env, pi, V, gamma):
    for s in range(env.nS):
        q_s = np.zeros([env.nA, 1])
        for a in range(env.nA):
            q_s[a] = compute_q_from_s_a(env, V, s, a, gamma)

        best_a = np.argmax(q_s)
        pi[s] = np.eye(env.nA)[best_a]
    return pi

pi = np.ones([grid.nS, grid.nA]) * 0.5
V = np.zeros([grid.nS, 1])
gamma = 0.99
theta = 0.0001

i = 0
while True:
    i += 1
    V, improvement = policy_evaluation(grid, pi, V, gamma, theta)
    pi = improve_policy(grid, pi, V, gamma)
    if not improvement:
        break

print("Optimal Policy:", pi)

