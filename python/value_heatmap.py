import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_value_heatmap(network, env, width: int, height: int) -> None:
        value_grid = np.zeros((width, height))
        for x in range(width):
            for y in range(height):
                state = [x, y]
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

