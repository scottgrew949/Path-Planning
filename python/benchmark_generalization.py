# python/benchmark_generalization.py
#
# Generalization study for the Neural A* learned heuristic.
#
# Research question:
#   What did the network actually learn — maze-specific patterns, or generalizable
#   geometric structure? We test the same weights.bin (trained on 41x41 labyrinth
#   mazes at density=0.25) against:
#
#   1. Grid size transfer: 21x21, 31x31, 41x41 (training), 61x61, 81x81
#      The network uses normalized coordinates [x/W, y/H] — does that enable
#      transfer to unseen scales, or is maze topology scale-dependent?
#
#   2. Maze type transfer: labyrinth (training) vs random obstacle placement
#      Labyrinth = connected corridors via recursive backtracking.
#      Random    = independent Bernoulli obstacle per cell (no connectivity guarantee).
#      If the network learned corridor structure, it should fail on random mazes.
#
# Interpretation guide:
#   Node reduction maintained across sizes → network learned geometry, not scale
#   Node reduction drops on random mazes   → network learned labyrinth structure
#   Node reduction drops on larger grids   → network overfit to 41x41 scale
#
# Run: source venv/bin/activate && python python/benchmark_generalization.py

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pathplanning

WEIGHTS_PATH     = os.path.join(os.path.dirname(__file__), 'data', 'weights.bin')
TRAINING_SIZE    = 41
TRAINING_DENSITY = 0.25
EPSILON          = 1.5    # same as primary benchmark
NUM_MAZES        = 500    # per condition — enough for statistical stability
TEST_SEED_OFFSET = 2000   # offset from training (0-499) and primary benchmark (1000-1999)

GRID_SIZES = [21, 31, 41, 61, 81]
MAZE_TYPES = ["labyrinth", "random"]


def run_pair(env, neural_algo, goal_x, goal_y):
    """Run both planners on the same env. Returns (astar_result, neural_result)."""
    start_x, start_y = 1, 1

    astar_path  = env.findPath(start_x, start_y, goal_x, goal_y)
    astar_nodes = env.getNodesExplored()
    if not astar_path:
        return None, None

    neural_path  = env.runNeuralAStar(neural_algo, start_x, start_y, goal_x, goal_y)
    neural_nodes = neural_algo.getNodesExplored()

    return (len(astar_path), astar_nodes), (len(neural_path) if neural_path else len(astar_path), neural_nodes)


def benchmark_size(size, maze_type, neural_algo):
    goal_x = size - 2
    goal_y = size - 2

    astar_nodes_list, neural_nodes_list   = [], []
    astar_path_list,  neural_path_list    = [], []
    skipped = 0

    for i in range(NUM_MAZES):
        seed = TEST_SEED_OFFSET + i
        env  = pathplanning.GridEnvironment(
            size, size, 1, 1, goal_x, goal_y,
            TRAINING_DENSITY, seed, maze_type
        )
        astar_result, neural_result = run_pair(env, neural_algo, goal_x, goal_y)
        if astar_result is None:
            skipped += 1
            continue

        astar_path_list.append(astar_result[0])
        astar_nodes_list.append(astar_result[1])
        neural_path_list.append(neural_result[0])
        neural_nodes_list.append(neural_result[1])

    if not astar_nodes_list:
        return None

    return {
        "astar_nodes_mean":  np.mean(astar_nodes_list),
        "astar_nodes_std":   np.std(astar_nodes_list),
        "neural_nodes_mean": np.mean(neural_nodes_list),
        "neural_nodes_std":  np.std(neural_nodes_list),
        "node_reduction":    (np.mean(astar_nodes_list) - np.mean(neural_nodes_list))
                             / np.mean(astar_nodes_list) * 100,
        "path_ratio":        np.mean(neural_path_list) / np.mean(astar_path_list),
        "solved":            len(astar_nodes_list),
        "skipped":           skipped,
    }


def print_size_table(results, maze_type):
    print(f"\n{'─'*80}")
    print(f"  Size generalization  |  maze type = {maze_type}  |  ε = {EPSILON}  |  {NUM_MAZES} mazes each")
    if maze_type == "labyrinth":
        print(f"  Training distribution: {TRAINING_SIZE}×{TRAINING_SIZE} labyrinth, density = {TRAINING_DENSITY}")
    else:
        print(f"  Out-of-distribution: random obstacle placement (network never trained on this)")
    print(f"{'─'*80}")
    print(f"  {'grid size':>10}  {'A* nodes (mean±std)':>22}  {'Neural nodes (mean±std)':>25}  {'nodes saved':>12}  {'path overhead':>14}")
    print(f"  {'':─>10}  {'':─>22}  {'':─>25}  {'':─>12}  {'':─>14}")

    for size in GRID_SIZES:
        r = results.get(size)
        if r is None:
            print(f"  {f'{size}×{size}':>10}  {'no solvable mazes':>63}")
            continue
        overhead = (r["path_ratio"] - 1) * 100
        marker = "  ← training size" if size == TRAINING_SIZE else ""
        print(f"  {f'{size}×{size}':>10}  "
              f"{r['astar_nodes_mean']:>10.1f} ± {r['astar_nodes_std']:>6.1f}  "
              f"{r['neural_nodes_mean']:>12.1f} ± {r['neural_nodes_std']:>6.1f}  "
              f"{r['node_reduction']:>+11.1f}%  "
              f"{overhead:>+12.1f}%"
              f"{marker}")
    print()


def print_type_table(results_labyrinth, results_random):
    size = TRAINING_SIZE
    r_lab = results_labyrinth.get(size)
    r_ran = results_random.get(size)
    if not r_lab or not r_ran:
        return

    print(f"\n{'─'*80}")
    print(f"  Maze type transfer  |  {size}×{size} grid  |  density = {TRAINING_DENSITY}  |  ε = {EPSILON}")
    print(f"  Network trained on: labyrinth mazes only")
    print(f"{'─'*80}")
    print(f"  {'maze type':>12}  {'A* nodes':>22}  {'Neural nodes':>25}  {'nodes saved':>12}  {'path overhead':>14}")
    print(f"  {'':─>12}  {'':─>22}  {'':─>25}  {'':─>12}  {'':─>14}")

    for label, r in [("labyrinth", r_lab), ("random", r_ran)]:
        overhead = (r["path_ratio"] - 1) * 100
        note = "  (training type)" if label == "labyrinth" else "  (never seen)"
        print(f"  {label:>12}  "
              f"{r['astar_nodes_mean']:>10.1f} ± {r['astar_nodes_std']:>6.1f}  "
              f"{r['neural_nodes_mean']:>12.1f} ± {r['neural_nodes_std']:>6.1f}  "
              f"{r['node_reduction']:>+11.1f}%  "
              f"{overhead:>+12.1f}%"
              f"{note}")
    print()


def print_findings(lab_results, ran_results):
    print(f"\n{'═'*80}")
    print("  FINDINGS")
    print(f"{'═'*80}")

    # Size trend
    reductions = [(s, lab_results[s]["node_reduction"]) for s in GRID_SIZES if s in lab_results]
    training_reduction = next(r for s, r in reductions if s == TRAINING_SIZE)

    sizes_better = [s for s, r in reductions if s != TRAINING_SIZE and r >= training_reduction - 5]
    sizes_worse  = [s for s, r in reductions if s != TRAINING_SIZE and r <  training_reduction - 5]

    if sizes_worse:
        print(f"  Size transfer: reduction degrades at sizes {sizes_worse} — network shows scale dependence")
    else:
        print(f"  Size transfer: reduction holds across all tested sizes — network learned scale-invariant geometry")

    # Type transfer
    r41_lab = lab_results.get(TRAINING_SIZE)
    r41_ran = ran_results.get(TRAINING_SIZE)
    if r41_lab and r41_ran:
        gap = r41_lab["node_reduction"] - r41_ran["node_reduction"]
        if gap > 10:
            print(f"  Maze type: {gap:.1f}pp drop on random mazes — network learned labyrinth corridor structure")
        else:
            print(f"  Maze type: reduction holds on random mazes ({gap:.1f}pp difference) — learned general geometry")

    print(f"{'═'*80}\n")


def main():
    if not os.path.exists(WEIGHTS_PATH):
        print("weights.bin not found. Run: ./build.sh 12")
        return

    neural_algo = pathplanning.NeuralAStar(WEIGHTS_PATH, EPSILON)
    print(f"\n{'═'*80}")
    print("  Neural A* Generalization Study")
    print(f"  Weights trained on: {TRAINING_SIZE}×{TRAINING_SIZE} labyrinth, density = {TRAINING_DENSITY}")
    print(f"  ε = {EPSILON}  |  {NUM_MAZES} mazes per condition")
    print(f"{'═'*80}")

    total     = len(GRID_SIZES) * len(MAZE_TYPES)
    done      = 0
    lab_results = {}
    ran_results = {}

    print(f"\n  Running {total} conditions...")
    print(f"  {'':─<60}")

    for maze_type in MAZE_TYPES:
        for size in GRID_SIZES:
            done += 1
            print(f"  [{done:>2}/{total}]  {size:>2}×{size:<2}  maze = {maze_type:<10}", end="", flush=True)
            start = time.time()
            r = benchmark_size(size, maze_type, neural_algo)
            elapsed = time.time() - start
            if r:
                store = lab_results if maze_type == "labyrinth" else ran_results
                store[size] = r
                print(f"  {elapsed:>4.1f}s    {r['solved']}/{NUM_MAZES} solved   reduction = {r['node_reduction']:+.1f}%")
            else:
                print(f"  {elapsed:>4.1f}s    no solvable mazes")

    print_size_table(lab_results, "labyrinth")
    print_size_table(ran_results, "random")
    print_type_table(lab_results, ran_results)
    print_findings(lab_results, ran_results)


if __name__ == "__main__":
    main()
