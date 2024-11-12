import RowGridWorld
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

grid = RowGridWorld.GridWorld(size=5)

# Initial policy (uniform random policy)
pi = np.ones((grid.nS, grid.nA)) * 0.5

# Given value function V (from policy evaluation)
V = np.array([0, -71.6, -43.7, -16.6, 0])  # V is now a 1D array
gamma = 0.99
theta = 0.0001

def compute_q_from_s_a(env, V, s, a, gamma):
    q = 0
    for prob, next_state, reward, done in env.P[s][a]:
        q += prob * (reward + gamma * V[next_state])
    return q

# Initialize Q(s, a) array
Q = np.zeros((grid.nS, grid.nA))

# Policy Improvement Step
for s in range(grid.nS):
    q_s = np.zeros(grid.nA)  # q_s is now a 1D array
    for a in range(grid.nA):
        q_s[a] = compute_q_from_s_a(grid, V, s, a, gamma)
        Q[s, a] = q_s[a]  # Store Q(s, a) for plotting

    best_a = np.argmax(q_s)
    pi[s] = np.eye(grid.nA)[best_a]

print("Optimal Policy:")
print(pi)

# Plotting the Updated Policy
# This plot shows the action chosen in each state after policy improvement.
# It helps visualize the agent's behavior under the new policy.
def plot_policy(grid, pi):
    nS = grid.nS
    fig, ax = plt.subplots(figsize=(nS, 2))
    action_symbols = {0: '←', 1: '→'}

    # Define colors for different state types
    colors = {
        'start': 'lightcoral',
        'goal': 'lightgreen',
        'intermediate': 'lightblue'
    }

    for s in range(nS):
        # Determine the state type
        if s == 0:
            state_type = 'start'
        elif s == nS - 1:
            state_type = 'goal'
        else:
            state_type = 'intermediate'

        # Draw state cell
        rect = patches.Rectangle((s, 0), 1, 1, linewidth=1, edgecolor='black',
                                 facecolor=colors[state_type])
        ax.add_patch(rect)

        # Add state number
        ax.text(s + 0.5, 0.8, f'S{s}', ha='center', va='center',
                fontsize=12, fontweight='bold')

        # Add policy action
        if state_type == 'intermediate':
            action = np.argmax(pi[s])
            action_symbol = action_symbols[action]
            ax.text(s + 0.5, 0.4, f'{action_symbol}', ha='center', va='center',
                    fontsize=20)
        else:
            ax.text(s + 0.5, 0.4, 'Terminal', ha='center', va='center',
                    fontsize=10)

    ax.set_xlim(0, nS)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Optimal Policy Visualization')
    plt.tight_layout()
    plt.show()

# Plotting the Action-Value Function Q(s, a)
# This plot shows the Q-values for each action in each state.
# It helps understand why certain actions are preferred in policy improvement.
def plot_action_values(Q):
    nS, nA = Q.shape
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(nS)
    width = 0.35
    ax.bar(x - width/2, Q[:, 0], width, label='Action 0 (←)')
    ax.bar(x + width/2, Q[:, 1], width, label='Action 1 (→)')
    ax.set_xlabel('States')
    ax.set_ylabel('Q(s, a)')
    ax.set_title('Action-Value Function')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in range(nS)])
    ax.legend()
    plt.show()

# Plotting the Value Function V(s)
# This plot shows the estimated value of each state.
# It helps to see how state values influence policy decisions.
def plot_value_function(grid, V):
    nS = grid.nS
    V = V.flatten()  # Ensure V is a 1D array
    fig, ax = plt.subplots(figsize=(nS, 2))

    # Define colors for different types of states
    colors = {
        'start': 'lightcoral',
        'goal': 'lightgreen',
        'intermediate': 'lightblue'
    }

    # Draw the cells for each state and annotate with value
    for s in range(nS):
        # Determine the state type
        if s == 0:
            state_type = 'start'
        elif s == nS - 1:
            state_type = 'goal'
        else:
            state_type = 'intermediate'

        # Create a square patch for the state
        rect = patches.Rectangle((s, 0), 1, 1, linewidth=1,
                                 edgecolor='black', facecolor=colors[state_type])
        ax.add_patch(rect)

        # Add state number and value
        ax.text(s + 0.5, 0.7, f'S{s}', ha='center', va='center',
                fontsize=12, fontweight='bold')
        ax.text(s + 0.5, 0.4, f'V={V[s]:.2f}', ha='center',
                va='center', fontsize=10)

    # Set limits and remove axes
    ax.set_xlim(0, nS)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('Value Function')
    plt.tight_layout()
    plt.show()

# Plotting the Optimal Policy
plot_policy(grid, pi)

# Plotting the Action-Value Function
plot_action_values(Q)

# Plotting the Value Function
plot_value_function(grid, V)
