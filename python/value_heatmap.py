import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_value_heatmap(network, env, width: int, height: int,
                       goal_x: int = None, goal_y: int = None) -> None:
    # goal_x/goal_y: required when network is goal-conditioned (STATE_SIZE=8).
    # Omit for standard DQN (STATE_SIZE=6).
    goal_conditioned = goal_x is not None and goal_y is not None
    value_grid = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            if goal_conditioned:
                normalised = [
                    x / width, y / height,
                    goal_x / width, goal_y / height,
                ] + list(env.getLineOfSight(x, y))
            else:
                normalised = [x / width, y / height] + list(env.getLineOfSight(x, y))

            state_tensor = torch.FloatTensor(normalised).unsqueeze(0)
            with torch.no_grad():
                q_values = network(state_tensor)
            value_grid[x][y] = q_values.max().item()

    plt.figure(figsize=(8, 8))
    plt.imshow(value_grid.T, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Max Q-value')
    plt.title('Value Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()
