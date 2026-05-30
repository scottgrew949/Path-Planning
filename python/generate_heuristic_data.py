# python/generate_heuristic_data.py
#
# PURPOSE: Generate supervised training data for the Neural A* heuristic network.
#
# CORE CONCEPT — What is h*(n)?
#   In A*, the heuristic h(n) estimates the cost from node n to the goal.
#   The PERFECT heuristic h*(n) is the TRUE remaining cost — the actual shortest
#   path length from n to the goal. If we could evaluate h* instantly, A* would
#   expand zero unnecessary nodes (it would follow the optimal path exactly).
#   We are going to LEARN an approximation of h* from data.
#
# CORE CONCEPT — Why not collect training data from A* paths?
#   Naively: run A*, record every node on the path, label it with distance to goal.
#   Problem: A* paths cluster along optimal corridors. Cells near walls, dead ends,
#   and "off-path" regions are almost never seen. A network trained on this data
#   would have massive gaps in its understanding of the heuristic landscape.
#   This is called DISTRIBUTION SHIFT — train on one distribution, fail on another.
#
# CORE CONCEPT — Backward Dijkstra (the right approach)
#   Instead: start at the GOAL and run Dijkstra BACKWARDS through the grid.
#   Every cell the search reaches gets labelled with its true cost to the goal.
#   One backward pass covers EVERY reachable cell uniformly in O(V) time.
#   This is the same idea as "value iteration" in RL — sweep costs backward
#   from the terminal state. Here we do it exactly with Dijkstra.
#
# CORE CONCEPT — Feature normalization
#   Raw grid coordinates like (x=37, y=12) are scale-dependent.
#   Different grid sizes would produce different input ranges, breaking the network.
#   Normalising to [0, 1] by dividing by width/height makes the features
#   DIMENSIONLESS — the network learns relative position, not absolute pixels.
#   Same reason: we normalise the h* label by (width + height) so output is in [0,1].
#   This keeps gradients well-conditioned during training (no exploding/vanishing).
#
# OUTPUT: python/data/heuristic_training.npy
#   Shape: (N, 5) — columns: [curr_x_norm, curr_y_norm, goal_x_norm, goal_y_norm, h_star_norm]

import heapq
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pathplanning  # pybind11 module — provides Environment, Position

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'heuristic_training.npy')

NUM_MAZES     = 500   # number of random environments to generate
GRID_WIDTH    = 41
GRID_HEIGHT   = 41
OBSTACLE_DENSITY = 0.25


def get_grid_neighbors(env, x: int, y: int) -> list:
    """
    CONCEPT — Graph adjacency:
    The grid IS a graph. Cells are vertices. Valid cardinal moves are edges.
    'getNeighbors' in C++ returns the same set — we replicate it here to
    avoid a pybind11 call inside the inner loop of Dijkstra (crossing the
    FFI boundary per node expansion is expensive).
    Implement: return [(nx, ny), ...] for the 4 cardinal directions,
    filtering to in-bounds, non-obstacle cells.
    """
    neighbors = []
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if env.isValid(nx, ny):
            neighbors.append((nx, ny))
    return neighbors


def backward_dijkstra(env, goal_x: int, goal_y: int) -> dict:
    """
    CONCEPT — Backward sweep / cost-to-go computation:
    Forward Dijkstra finds shortest path FROM source TO all destinations.
    Backward Dijkstra runs on the SAME graph but starts from the goal and
    finds the shortest path FROM every cell TO the goal.
    For undirected, uniform-cost graphs (our cardinal grid), these are identical.
    For directed graphs (road networks with one-way streets), they differ.

    The result is the COMPLETE h* landscape for this maze in one O(V log V) pass.
    Compare: running A* for every cell separately would be O(V * V log V).

    CONCEPT — Priority queue / min-heap:
    heapq in Python is a min-heap. We push (cost, x, y) tuples.
    The heap property guarantees we always process the cheapest unvisited cell first.
    This is what makes Dijkstra correct — once a cell is popped, its distance is final.

    Implement:
    1. Initialize dist = {} (empty dict = all cells have cost infinity)
    2. Push (0, goal_x, goal_y) onto heap
    3. While heap not empty:
       a. Pop (cost, x, y)
       b. If (x,y) already in dist: skip (already settled with lower cost)
       c. dist[(x,y)] = cost
       d. For each neighbor (nx, ny): push (cost + 1, nx, ny)
    4. Return dist — maps (x,y) → true h* to goal
    """
    dist = {}
    heap = [(0, goal_x, goal_y)]
    while heap:
        cost, x, y = heapq.heappop(heap)
        if (x, y) in dist:
            continue
        dist[(x, y)] = cost
        for nx, ny in get_grid_neighbors(env, x, y):
            if (nx, ny) not in dist:
                heapq.heappush(heap, (cost + 1, nx, ny))
    return dist


def generate_samples_for_maze(env, width: int, height: int) -> np.ndarray:
    """
    CONCEPT — Supervised learning label generation:
    Supervised learning needs (input, label) pairs.
    Input:  [curr_x/W, curr_y/H, goal_x/W, goal_y/H]
    Label:  h*(curr, goal) / (W + H - 2)    ← normalised true cost-to-go

    We generate one set of samples per maze by:
    1. Picking a random goal position (non-obstacle, not at edge)
    2. Running backward Dijkstra from that goal
    3. Emitting one row per reachable cell

    Why (W + H - 2) as normaliser?
    That is the Manhattan distance corner-to-corner — the MAXIMUM possible h*
    in an obstacle-free grid. Normalising by this maps all h* values to [0, 1].

    Implement:
    1. Pick a random valid goal position using np.random.randint
    2. Run backward_dijkstra(env, goal_x, goal_y)
    3. For each (x,y) in dist: build one row [x/W, y/H, gx/W, gy/H, h*/max_h]
    4. Return np.array of shape (num_reachable_cells, 5)
    """
    max_h   = float(width + height - 2)

    for _ in range(200):
        goal_x = np.random.randint(1, width  - 1)
        goal_y = np.random.randint(1, height - 1)
        if env.isValid(goal_x, goal_y):
            break
    else:
        goal_x, goal_y = width // 2, height // 2

    dist = backward_dijkstra(env, goal_x, goal_y)
    if not dist:
        return np.empty((0, 5), dtype=np.float32)

    rows = []
    gx_norm = goal_x / width
    gy_norm = goal_y / height
    for (x, y), h_star in dist.items():
        rows.append([x / width, y / height, gx_norm, gy_norm, h_star / max_h])

    return np.array(rows, dtype=np.float32)


def main():
    """
    CONCEPT — Dataset construction pipeline:
    ML training quality is determined as much by data quality as by model architecture.
    Here we control three key properties:
      1. COVERAGE:  backward Dijkstra ensures every cell is sampled, not just path cells
      2. DIVERSITY: different random seeds produce different maze topologies
      3. BALANCE:   normalised features prevent any single dimension dominating

    Implement:
    1. Loop NUM_MAZES times with different random seeds
    2. Create Environment(GRID_WIDTH, GRID_HEIGHT), call generateRandom(OBSTACLE_DENSITY)
    3. Call generate_samples_for_maze(), accumulate into a list
    4. Concatenate all sample arrays: np.vstack(all_samples)
    5. Shuffle rows (random row order prevents the network from learning maze order)
    6. Save to OUTPUT_PATH with np.save()
    7. Print: total samples, feature min/max (sanity check normalisation)
    """
    all_samples = []

    for seed in range(NUM_MAZES):
        env     = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1,
                                               GRID_WIDTH - 2, GRID_HEIGHT - 2,
                                               OBSTACLE_DENSITY, seed)
        samples = generate_samples_for_maze(env, GRID_WIDTH, GRID_HEIGHT)
        if samples.shape[0] > 0:
            all_samples.append(samples)

        if (seed + 1) % 50 == 0:
            print(f"Processed {seed + 1}/{NUM_MAZES} mazes")

    dataset = np.vstack(all_samples)
    np.random.shuffle(dataset)
    np.save(OUTPUT_PATH, dataset)

    print(f"Saved {len(dataset):,} samples to {OUTPUT_PATH}")
    print(f"Feature min: {dataset.min(axis=0)}")
    print(f"Feature max: {dataset.max(axis=0)}")


if __name__ == '__main__':
    main()
