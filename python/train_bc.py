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

GRID_WIDTH    = 201
GRID_HEIGHT   = 41
STATE_SIZE    = 6
ACTION_SIZE   = 4
HIDDEN_SIZE   = 128
LEARNING_RATE = 1e-3

TRAIN_EPOCHS      = 2000
LOG_EVERY         = 200
N_EVAL_EPISODES   = 50
MAX_EVAL_STEPS    = 2000

# ---- Training loop ----------------------------------------------------------

def train():
    env = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 0, 0, 200, 40, 0.4, 42)
    goal = env.getGoal()
    gx, gy = goal[0], goal[1]

    # -- Step 1: collect expert trajectory ------------------------------------
    raw = env.getExpertTrajectory()   # flat list [x0,y0,a0, x1,y1,a1, ...]
    if len(raw) == 0:
        print("WARNING: expert trajectory is empty — no path found. Exiting.")
        return

    triples = [(int(raw[i]), int(raw[i + 1]), int(raw[i + 2]))
               for i in range(0, len(raw), 3)]

    # -- Step 2: build dataset ------------------------------------------------
    states  = torch.cat([encode_state(x, y, gx, gy, GRID_WIDTH, GRID_HEIGHT,
                                      env.getLineOfSight(x, y))
                         for x, y, _ in triples])
    actions = torch.LongTensor([a for _, _, a in triples])

    print(f"Expert trajectory: {len(triples)} steps")

    # -- Step 3: train --------------------------------------------------------
    agent = BCAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LEARNING_RATE)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        loss = agent.update(states, actions)
        if epoch % LOG_EVERY == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

    # -- Step 4: evaluate -----------------------------------------------------
    successes   = 0
    total_steps = 0

    for _ in range(N_EVAL_EPISODES):
        pos  = env.reset()   # [x, y]
        done = False
        for step in range(MAX_EVAL_STEPS):
            los     = env.getLineOfSight(int(pos[0]), int(pos[1]))
            state_t = encode_state(pos[0], pos[1], gx, gy, GRID_WIDTH, GRID_HEIGHT, los)
            action  = agent.select_action(state_t)
            result  = env.step(action)   # [new_x, new_y, reward, done_float]
            pos     = [result[0], result[1]]
            if result[3] > 0.5:
                successes   += 1
                total_steps += step + 1
                done = True
                break
        if not done:
            total_steps += MAX_EVAL_STEPS

    avg_steps = total_steps / successes if successes > 0 else float("nan")
    print(f"BC success rate: {successes}/{N_EVAL_EPISODES}")
    print(f"Average steps to goal: {avg_steps:.1f}")

    # -- Step 5: save ---------------------------------------------------------
    torch.save(agent.network.state_dict(), "bc_model.pth")
    print("Saved bc_model.pth")


if __name__ == "__main__":
    train()
