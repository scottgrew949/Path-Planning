# python/train_dagger.py
# DAgger (Dataset Aggregation) — iterative imitation learning with expert relabeling.
#
# CONCEPT — BC vs DAgger
#   Behavioral Cloning trains only on the expert's trajectory — states the expert
#   visited. At test time the agent drifts off-path and encounters novel states
#   the policy was never trained on. Errors compound: one bad step leads to a
#   worse state, which causes another bad step.
#
#   DAgger breaks this cycle iteratively:
#     1. Roll out the CURRENT policy in the environment.
#     2. For every state the agent visits, ask the EXPERT what action it would take.
#     3. Add those (state, expert_action) pairs to the growing dataset.
#     4. Retrain the policy on the full dataset.
#   Repeat. Each iteration the policy sees more of the state space it actually
#   visits at test time, including recovery states after mistakes.
#   Over iterations the policy learns to correct its own errors.
#
# CONCEPT — Why keep the full growing dataset?
#   Discarding old data would let the policy forget recovery paths it learned
#   in earlier iterations. Aggregation (the "Ag" in DAgger) means every expert
#   label collected so far stays in the training set — monotonically growing.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pathplanning
from agents.bc_agent import BCAgent, encode_state

# ---- Hyperparameters --------------------------------------------------------

GRID_WIDTH    = 41
GRID_HEIGHT   = 41
GOAL_X        = 38
GOAL_Y        = 40
LOOP_DENSITY  = 0.3
STATE_SIZE    = 6
ACTION_SIZE   = 4
HIDDEN_SIZE   = 128
LEARNING_RATE = 1e-3

N_TRAIN_MAZES       = 100  # seed pool for training rollouts
N_DAGGER_ITERATIONS = 10
N_EPOCHS_PER_ITER   = 500
MAX_ROLLOUT_STEPS   = 500
N_EVAL_EPISODES     = 50   # held-out seeds per iteration eval
ROLLOUT_EPISODES    = 20   # maze rollouts collected per DAgger iteration

# ---- Training loop ----------------------------------------------------------

def train():
    agent = BCAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LEARNING_RATE)

    try:
        agent.network.load_state_dict(torch.load("bc_model.pth", weights_only=True))
        print("Loaded bc_model.pth as starting point.")
    except FileNotFoundError:
        print("bc_model.pth not found — training from scratch (run train_bc.py first for best results).")

    states_list  = []
    actions_list = []

    for iteration in range(1, N_DAGGER_ITERATIONS + 1):

        # -- Collect data -----------------------------------------------------
        if iteration == 1:
            # Seed dataset with expert trajectories from all training mazes.
            print(f"Iter  1 | seeding dataset from {N_TRAIN_MAZES} mazes...")
            for seed in range(N_TRAIN_MAZES):
                env  = pathplanning.GridEnvironment(
                    GRID_WIDTH, GRID_HEIGHT, 0, 0, GOAL_X, GOAL_Y, LOOP_DENSITY, seed)
                goal = env.getGoal()
                gx, gy = int(goal[0]), int(goal[1])
                raw = env.getExpertTrajectory()
                if len(raw) == 0:
                    continue
                for i in range(0, len(raw), 3):
                    x, y, a = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
                    los = env.getLineOfSight(x, y)
                    states_list.append(encode_state(x, y, gx, gy, GRID_WIDTH, GRID_HEIGHT, los))
                    actions_list.append(a)
        else:
            # Roll out current policy on rotating maze seeds; label with expert.
            maze_seeds = [
                (iteration * ROLLOUT_EPISODES + r) % N_TRAIN_MAZES
                for r in range(ROLLOUT_EPISODES)
            ]
            for seed in maze_seeds:
                env  = pathplanning.GridEnvironment(
                    GRID_WIDTH, GRID_HEIGHT, 0, 0, GOAL_X, GOAL_Y, LOOP_DENSITY, seed)
                goal = env.getGoal()
                gx, gy = int(goal[0]), int(goal[1])
                pos = env.reset()
                for _ in range(MAX_ROLLOUT_STEPS):
                    los           = env.getLineOfSight(int(pos[0]), int(pos[1]))
                    state_t       = encode_state(pos[0], pos[1], gx, gy, GRID_WIDTH, GRID_HEIGHT, los)
                    action        = agent.select_action(state_t)
                    result        = env.step(action)
                    expert_action = env.getExpertAction(int(pos[0]), int(pos[1]))
                    if expert_action >= 0:
                        states_list.append(encode_state(pos[0], pos[1], gx, gy,
                                                        GRID_WIDTH, GRID_HEIGHT, los))
                        actions_list.append(expert_action)
                    pos = [result[0], result[1]]
                    if result[3] > 0.5:
                        break

        # -- Train on full aggregated dataset ---------------------------------
        states_tensor  = torch.cat(states_list)
        actions_tensor = torch.LongTensor(actions_list)

        last_loss = 0.0
        for _ in range(N_EPOCHS_PER_ITER):
            last_loss = agent.update(states_tensor, actions_tensor)

        print(f"Iter {iteration:2d} | dataset: {len(actions_list):5d} samples | loss: {last_loss:.4f}")

        # -- Evaluate on held-out seeds (never seen during training) ----------
        successes   = 0
        total_steps = 0
        eval_seeds  = range(N_TRAIN_MAZES, N_TRAIN_MAZES + N_EVAL_EPISODES)

        for seed in eval_seeds:
            env  = pathplanning.GridEnvironment(
                GRID_WIDTH, GRID_HEIGHT, 0, 0, GOAL_X, GOAL_Y, LOOP_DENSITY, seed)
            goal = env.getGoal()
            gx, gy = int(goal[0]), int(goal[1])
            pos  = env.reset()
            done = False
            for step in range(MAX_ROLLOUT_STEPS):
                los     = env.getLineOfSight(int(pos[0]), int(pos[1]))
                state_t = encode_state(pos[0], pos[1], gx, gy, GRID_WIDTH, GRID_HEIGHT, los)
                action  = agent.select_action(state_t)
                result  = env.step(action)
                pos     = [result[0], result[1]]
                if result[3] > 0.5:
                    successes   += 1
                    total_steps += step + 1
                    done = True
                    break
            if not done:
                total_steps += MAX_ROLLOUT_STEPS

        avg_steps = total_steps / successes if successes > 0 else float("nan")
        print(f"Iter {iteration:2d} | success (held-out): {successes}/{N_EVAL_EPISODES} | avg steps: {avg_steps:.1f}")

    # -- Save -----------------------------------------------------------------
    torch.save(agent.network.state_dict(), "dagger_model.pth")
    print("Saved dagger_model.pth")


if __name__ == "__main__":
    train()
