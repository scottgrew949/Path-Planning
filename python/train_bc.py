# python/train_bc.py
# Behavioral Cloning training loop — supervised learning from expert trajectory.
#
# CONCEPT — Training procedure
#   1. Ask the C++ A* planner for a complete optimal trajectory (getExpertTrajectory).
#   2. Convert that trajectory into a fixed (states, actions) dataset.
#   3. Train BCAgent on the dataset for many epochs — no environment interaction.
#   4. Evaluate the trained policy by rolling it out in the live environment.
#
# CONCEPT — Why evaluate after pure supervised training?
#   Loss going down does not guarantee the policy navigates correctly — the
#   network might overfit or the expert path might be too short to cover all
#   recoverable states. Rollout success rate is the real metric.

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

N_TRAIN_MAZES     = 100   # diverse maze seeds for training
N_EVAL_MAZES      = 50    # held-out seeds (never seen during training)
TRAIN_EPOCHS      = 2000
LOG_EVERY         = 200
MAX_EVAL_STEPS    = 500

# ---- Training loop ----------------------------------------------------------

def train():
    states_list  = []
    actions_list = []

    # -- Step 1: collect expert trajectories from N diverse mazes -------------
    print(f"Collecting expert trajectories from {N_TRAIN_MAZES} mazes...")
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
            states_list.append(encode_state(x, y, gx, gy, GRID_WIDTH, GRID_HEIGHT,
                                            env.getLineOfSight(x, y)))
            actions_list.append(a)

        if (seed + 1) % 20 == 0:
            print(f"  {seed + 1}/{N_TRAIN_MAZES} mazes — {len(actions_list)} samples")

    print(f"Dataset: {len(actions_list)} (state, action) pairs")

    # -- Step 2: train --------------------------------------------------------
    states  = torch.cat(states_list)
    actions = torch.LongTensor(actions_list)
    agent   = BCAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LEARNING_RATE)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        loss = agent.update(states, actions)
        if epoch % LOG_EVERY == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

    # -- Step 3: evaluate on held-out seeds (agent has never seen these mazes)
    successes   = 0
    total_steps = 0
    eval_seeds  = range(N_TRAIN_MAZES, N_TRAIN_MAZES + N_EVAL_MAZES)

    for seed in eval_seeds:
        env  = pathplanning.GridEnvironment(
            GRID_WIDTH, GRID_HEIGHT, 0, 0, GOAL_X, GOAL_Y, LOOP_DENSITY, seed)
        goal = env.getGoal()
        gx, gy = int(goal[0]), int(goal[1])
        pos  = env.reset()
        done = False
        for step in range(MAX_EVAL_STEPS):
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
            total_steps += MAX_EVAL_STEPS

    avg_steps = total_steps / successes if successes > 0 else float("nan")
    print(f"BC success rate (held-out mazes): {successes}/{N_EVAL_MAZES}")
    print(f"Average steps to goal: {avg_steps:.1f}")

    # -- Step 4: save ---------------------------------------------------------
    torch.save(agent.network.state_dict(), "bc_model.pth")
    print("Saved bc_model.pth")


if __name__ == "__main__":
    train()
