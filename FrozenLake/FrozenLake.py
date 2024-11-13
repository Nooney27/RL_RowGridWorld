import gym
import numpy as np


env = gym.make('FrozenLake-v1', render_mode="human")
#env = gym.make("CliffWalking-v0", render_mode="human")
#print(env.P)

def compute_q_value_for_s_a(env, V, s, a, gamma):
    q = 0
    for (p_s_prime, s_prime, r, is_terminal) in env.P[s][a]:
        q += p_s_prime * (r + gamma * V[s_prime])
    return q

def evaluate_policy(env, pi, V, gamma, theta):
    v_updated = np.copy(V)
    improved = True

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v_new = 0
            for a in range(env.action_space.n):
                prob_a = pi[s][a]
                q_s_a = compute_q_value_for_s_a(env, V, s, a, gamma)
                v_new += prob_a * q_s_a
            delta = max(delta, np.abs(v_updated[s] - v_new))
            v_updated[s] = v_new
        if delta < theta:
            break
    if np.array_equal(V, v_updated):
        improved = False
    
    return v_updated, improved

def improve_policy(env, pi, V, gamma):
    for s in range(env.observation_space.n):
        q_s = np.zeros([env.action_space.n, 1])
        for a in range(env.action_space.n):
            q_s[a] = compute_q_value_for_s_a(env, V, s, a, gamma)
        best_a = np.argmax(q_s)
        pi[s] = np.eye(env.action_space.n)[best_a]
    return pi

pi = np.ones([env.observation_space.n, env.action_space.n]) * 0.5
V = np.zeros([env.observation_space.n, 1])
gamma = 0.99
theta = 1e-8

i = 0
while True:
    i += 1
    V, improved = evaluate_policy(env, pi, V, gamma, theta)
    pi = improve_policy(env, pi, V, gamma)

    if improved == False:
        print('Converged after {} iterations'.format(i))
        break
print("Optimal policy: ", pi)

def print_policy(env, pi):
    policy_arrows = {
        0: '←',  # LEFT
        1: '↓',  # DOWN
        2: '→',  # RIGHT
        3: '↑'   # UP
    }
    
    policy_grid = []
    for s in range(env.observation_space.n):
        best_a = np.argmax(pi[s])
        policy_grid.append(policy_arrows[best_a])
    
    policy_grid = np.array(policy_grid).reshape(env.desc.shape)
    for row in policy_grid:
        print(' '.join(row))

print_policy(env, pi)

# Visualize the optimal policy in the environment
state, info = env.reset()
done = False

print("\nExecuting the optimal policy:\n")
while not done:
    state_int = int(state)  # Ensure state is an integer
    action = np.argmax(pi[state_int])
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
    if done:
        if reward == 1.0:
            print("Reached the goal!")
        else:
            print("Fell into a hole!")
        break