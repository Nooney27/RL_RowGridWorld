# RowGridWorld.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx

class GridWorld():
    def __init__(self, size=5):
        self.size = size
        self.nS = size  # number of states
        self.nA = 2     # number of actions
        self.MAX_X = size - 1  # state maximal for the agent (state 4)

        P = {}
        for s in range(self.nS):
            dynamics_s = {}
            for a in range(self.nA):
                s_prime_list = []
                p = 1 if s != 0 and s != self.nS - 1 else 0

                if a == 0:
                    s_prime = max(0, s - 1)
                else:
                    s_prime = min(self.MAX_X, s + 1)

                if s_prime == self.MAX_X:
                    r = 10
                    done = True
                elif s_prime == 0:
                    r = -100
                    done = True
                else:
                    r = 0
                    done = False

                s_prime_list.append((p, s_prime, r, done))
                dynamics_s.update({a: s_prime_list})
            P.update({s: dynamics_s})

        self.P = P

    def plot_gridworld(self):
        nS = self.nS
        fig, ax = plt.subplots(figsize=(nS, 2))

        # Define colors for different types of states
        colors = {
            'start': 'lightcoral',
            'goal': 'lightgreen',
            'intermediate': 'lightblue'
        }

        # Draw the cells for each state
        for s in range(nS):
            # Determine the state type
            if s == 0:
                state_type = 'start'
            elif s == nS - 1:
                state_type = 'goal'
            else:
                state_type = 'intermediate'

            # Create a square patch for the state
            rect = patches.Rectangle((s, 0), 1, 1, linewidth=1, edgecolor='black',
                                     facecolor=colors[state_type])
            ax.add_patch(rect)

            # Add state number
            ax.text(s + 0.5, 0.7, f'S{s}', ha='center', va='center',
                    fontsize=12, fontweight='bold')

            # Add reward information
            if state_type == 'start':
                ax.text(s + 0.5, 0.4, f'Reward\n-100', ha='center',
                        va='center', fontsize=8)
            elif state_type == 'goal':
                ax.text(s + 0.5, 0.4, f'Reward\n+10', ha='center',
                        va='center', fontsize=8)
            else:
                ax.text(s + 0.5, 0.4, 'Reward\n0', ha='center',
                        va='center', fontsize=8)

            # Draw arrows for possible actions
            for a in range(self.nA):
                transitions = self.P[s][a]
                for (p, s_prime, r, done) in transitions:
                    if p > 0 and s != s_prime:
                        # Calculate arrow properties
                        dx = s_prime - s
                        if dx == -1:
                            # Left action
                            ax.arrow(s + 0.5, 0.2, -0.4, 0, head_width=0.05,
                                     head_length=0.05, fc='k', ec='k')
                        elif dx == 1:
                            # Right action
                            ax.arrow(s + 0.5, 0.2, 0.4, 0, head_width=0.05,
                                     head_length=0.05, fc='k', ec='k')

        # Set limits and remove axes
        ax.set_xlim(0, nS)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def display_transition_table(self):
        data = []
        for s in self.P:
            for a in self.P[s]:
                transitions = self.P[s][a]
                for (p, s_prime, r, done) in transitions:
                    data.append({
                        'State': f'S{s}',
                        'Action': f'A{a}',
                        'Next State': f'S{s_prime}',
                        'Probability': p,
                        'Reward': r,
                        'Done': done
                    })
        df = pd.DataFrame(data)
        print(df.to_string(index=False))

# Additional plotting functions

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
    plt.title('Value Function after Convergence')
    plt.tight_layout()
    plt.show()

def plot_value_convergence(V_history):
    V_history = np.array(V_history)
    iterations = np.arange(len(V_history))
    for s in range(V_history.shape[1]):
        plt.plot(iterations, V_history[:, s], label=f'State {s}')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Value Function Convergence')
    plt.legend()
    plt.show()

def plot_deltas(deltas):
    plt.figure()
    plt.plot(deltas)
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.title('Value Function Convergence (Delta over Iterations)')
    plt.yscale('log')
    plt.show()

def plot_value_heatmap(V):
    V = V.flatten()
    plt.figure(figsize=(8, 1))
    sns.heatmap([V], annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
                xticklabels=[f'S{s}' for s in range(len(V))], yticklabels=[])
    plt.title('Value Function Heatmap')
    plt.show()


if __name__ == "__main__":
    env = GridWorld()
    env.plot_gridworld()
    env.display_transition_table()
