# python/benchmark_imitation.py
# Side-by-side comparison of all imitation learning policies on the same maze.
#
# CONCEPT — Why benchmark on the same fixed env?
#   Each policy sees identical obstacle layout, start, and goal — differences in
#   success rate and path length are due to the policy, not maze randomness.
#   Random and Expert form the floor/ceiling; BC and DAgger sit between them.
#
# Policies:
#   Random  — uniform random action, no learning.  Expected success near 0%.
#   Expert  — A* oracle via getExpertAction.        Should hit 100% by definition.
#   BC      — trained by train_bc.py (bc_model.pth).
#   DAgger  — trained by train_dagger.py (dagger_model.pth).

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import pathplanning
from agents.bc_agent import BCAgent, encode_state

# ---- Hyperparameters --------------------------------------------------------

GRID_WIDTH    = 201
GRID_HEIGHT   = 41
STATE_SIZE    = 6
ACTION_SIZE   = 4
HIDDEN_SIZE   = 128

N_BENCHMARK_EPISODES = 100
MAX_STEPS            = 2000

# ---- Policy wrappers --------------------------------------------------------

def rollout_policy(env, policy_fn, n_episodes, max_steps, gx, gy):
    """Run n_episodes rollouts and return (successes, steps_list_on_success)."""
    successes  = 0
    steps_list = []
    for _ in range(n_episodes):
        pos = env.reset()   # [x, y]
        for step in range(max_steps):
            los    = env.getLineOfSight(int(pos[0]), int(pos[1]))
            action = policy_fn(pos[0], pos[1], gx, gy, los)
            result = env.step(action)   # [new_x, new_y, reward, done_float]
            pos    = [result[0], result[1]]
            if result[3] > 0.5:
                successes += 1
                steps_list.append(step + 1)
                break
    return successes, steps_list

# ---- Main -------------------------------------------------------------------

def main():
    env  = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 0, 0, 200, 40, 0.4, 42)
    goal = env.getGoal()
    gx, gy = goal[0], goal[1]

    results = {}

    # -- Random policy --------------------------------------------------------
    def random_policy(x, y, gx, gy, los):
        return random.randint(0, ACTION_SIZE - 1)

    s, steps = rollout_policy(env, random_policy,
                              N_BENCHMARK_EPISODES, MAX_STEPS, gx, gy)
    results["Random"] = (s, steps, "no learning")

    # -- Expert policy (A* oracle) --------------------------------------------
    def expert_policy(x, y, gx, gy, los):
        a = env.getExpertAction(int(x), int(y))
        if a < 0:
            return random.randint(0, ACTION_SIZE - 1)   # fallback; should not happen
        return a

    s, steps = rollout_policy(env, expert_policy,
                              N_BENCHMARK_EPISODES, MAX_STEPS, gx, gy)
    results["Expert"] = (s, steps, "A* oracle")

    # -- BC policy ------------------------------------------------------------
    bc_agent = BCAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    try:
        bc_agent.network.load_state_dict(torch.load("bc_model.pth", weights_only=True))
        bc_agent.network.eval()

        def bc_policy(x, y, gx, gy, los):
            state_t = encode_state(x, y, gx, gy, GRID_WIDTH, GRID_HEIGHT, los)
            return bc_agent.select_action(state_t)

        s, steps = rollout_policy(env, bc_policy,
                                  N_BENCHMARK_EPISODES, MAX_STEPS, gx, gy)
        results["BC"] = (s, steps, "supervised")
    except FileNotFoundError:
        print("bc_model.pth not found — run train_bc.py first.")
        results["BC"] = (None, [], "not available")

    # -- DAgger policy --------------------------------------------------------
    dagger_agent = BCAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    try:
        dagger_agent.network.load_state_dict(
            torch.load("dagger_model.pth", weights_only=True))
        dagger_agent.network.eval()

        def dagger_policy(x, y, gx, gy, los):
            state_t = encode_state(x, y, gx, gy, GRID_WIDTH, GRID_HEIGHT, los)
            return dagger_agent.select_action(state_t)

        s, steps = rollout_policy(env, dagger_policy,
                                  N_BENCHMARK_EPISODES, MAX_STEPS, gx, gy)
        results["DAgger"] = (s, steps, "iterative")
    except FileNotFoundError:
        print("dagger_model.pth not found — run train_dagger.py first.")
        results["DAgger"] = (None, [], "not available")

    # -- Print table ----------------------------------------------------------
    col_policy = 8
    col_rate   = 14
    col_steps  = 21
    col_notes  = 12

    header = (f"{'Policy':<{col_policy}} | "
              f"{'Success Rate':>{col_rate}} | "
              f"{'Avg Steps (success)':>{col_steps}} | "
              f"Notes")
    divider = "-" * len(header)
    print()
    print(header)
    print(divider)

    for name, (successes, steps_list, notes) in results.items():
        if successes is None:
            rate_str  = "N/A"
            steps_str = "N/A"
        else:
            pct       = 100.0 * successes / N_BENCHMARK_EPISODES
            rate_str  = f"{successes}/{N_BENCHMARK_EPISODES} ({pct:.0f}%)"
            steps_str = (f"{sum(steps_list)/len(steps_list):.1f}"
                         if steps_list else "N/A")
        print(f"{name:<{col_policy}} | "
              f"{rate_str:>{col_rate}} | "
              f"{steps_str:>{col_steps}} | "
              f"{notes}")

    print(divider)
    print()


if __name__ == "__main__":
    main()
