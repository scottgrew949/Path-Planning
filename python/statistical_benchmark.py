import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import random
import time
import torch
import pathplanning
from agents.bc_agent import BCAgent, encode_state

GRID_WIDTH  = 41
GRID_HEIGHT = 41
STATE_SIZE  = 6
ACTION_SIZE = 4
HIDDEN_SIZE = 128

START_X = 0
START_Y = 0
GOAL_X  = 38
GOAL_Y  = 40


def _build_env(difficulty, seed):
    return pathplanning.GridEnvironment(
        GRID_WIDTH, GRID_HEIGHT,
        START_X, START_Y, GOAL_X, GOAL_Y,
        difficulty, seed,
    )


def _run_episode(env, policy_fn, max_steps, gx, gy):
    pos = env.reset()
    x, y = int(pos[0]), int(pos[1])
    t0 = time.perf_counter()
    for step in range(max_steps):
        action = policy_fn(x, y, gx, gy, env)
        result = env.step(action)
        x, y = int(result[0]), int(result[1])
        if result[3] > 0.5:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return step + 1, step + 1, 1, elapsed_ms
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return -1, max_steps, 0, elapsed_ms


def _expert_policy(x, y, gx, gy, env):
    a = env.getExpertAction(x, y)
    if a < 0:
        return random.randint(0, ACTION_SIZE - 1)
    return a


def _random_policy(x, y, gx, gy, env):
    return random.randint(0, ACTION_SIZE - 1)


def _make_bc_policy(agent):
    def policy(x, y, gx, gy, env):
        los     = env.getLineOfSight(x, y)
        state_t = encode_state(x, y, gx, gy, GRID_WIDTH, GRID_HEIGHT, los)
        return agent.select_action(state_t)
    return policy


def _load_bc_agent(path, label):
    agent = BCAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    try:
        agent.network.load_state_dict(torch.load(path, weights_only=True))
        agent.network.eval()
        return agent
    except FileNotFoundError:
        print(f"WARNING: {path} not found — skipping {label}.")
        return None
    except Exception as e:
        print(f"WARNING: failed to load {path} ({e}) — skipping {label}.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",      type=int,   default=1000)
    parser.add_argument("--difficulty", type=float, default=0.3)
    parser.add_argument("--max-steps",  type=int,   default=500)
    parser.add_argument("--output",     type=str,   default="benchmark_results.csv")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path  = os.path.join(project_root, args.output)

    bc_agent     = _load_bc_agent(os.path.join(project_root, "bc_model.pth"),     "BC")
    dagger_agent = _load_bc_agent(os.path.join(project_root, "dagger_model.pth"), "DAgger")

    algorithms = [("Expert", _expert_policy), ("Random", _random_policy)]
    if bc_agent is not None:
        algorithms.append(("BC", _make_bc_policy(bc_agent)))
    if dagger_agent is not None:
        algorithms.append(("DAgger", _make_bc_policy(dagger_agent)))

    rows      = []
    successes = {name: 0 for name, _ in algorithms}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "algorithm", "path_length", "steps", "success", "time_ms", "difficulty"])

        for seed_idx in range(1, args.seeds + 1):
            env  = _build_env(args.difficulty, seed_idx)
            goal = env.getGoal()
            gx, gy = int(goal[0]), int(goal[1])

            for name, policy_fn in algorithms:
                path_length, steps, success, time_ms = _run_episode(
                    env, policy_fn, args.max_steps, gx, gy
                )
                if success:
                    successes[name] += 1
                row = [seed_idx, name, path_length, steps, success,
                       f"{time_ms:.3f}", args.difficulty]
                writer.writerow(row)
                rows.append(row)

            if seed_idx % 100 == 0:
                parts = []
                for name, _ in algorithms:
                    pct = 100.0 * successes[name] / seed_idx
                    parts.append(f"{name}: {pct:.0f}%")
                print(f"Seed {seed_idx}/{args.seeds} — {' | '.join(parts)}")

    print()
    col_w = max(len(name) for name, _ in algorithms) + 2

    header = (f"{'Algorithm':<{col_w}} {'Success Rate':>14} "
              f"{'Mean Path Len':>15} {'Mean Steps':>12}")
    print(header)
    print("-" * len(header))

    for name, _ in algorithms:
        algo_rows   = [r for r in rows if r[1] == name]
        n           = len(algo_rows)
        n_success   = sum(r[4] for r in algo_rows)
        rate        = 100.0 * n_success / n if n else 0.0
        success_rows = [r for r in algo_rows if r[4] == 1]
        mean_path   = (sum(r[2] for r in success_rows) / len(success_rows)
                       if success_rows else float("nan"))
        mean_steps  = sum(r[3] for r in algo_rows) / n if n else float("nan")
        print(f"{name:<{col_w}} {f'{n_success}/{n} ({rate:.1f}%)':>14} "
              f"{mean_path:>15.1f} {mean_steps:>12.1f}")

    print()
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
