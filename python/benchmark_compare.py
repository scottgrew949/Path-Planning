# python/benchmark_compare.py
# Head-to-head comparison: Tabular Q-Learning vs DQN on the same maze.
#
# CONCEPT — Why compare at all?
#   Having many algorithms means nothing if you can't answer "which one actually
#   learned better?" This script runs both on an identical grid for the same
#   number of episodes, then evaluates both policies greedily — same start,
#   same goal, same maze. The comparison reveals whether the added complexity
#   of DQN (neural net, replay buffer, target network) buys anything over a
#   simple lookup table on a 41x41 grid.
#
# CONCEPT — What to expect
#   On small grids (41x41 = 1681 states), tabular Q-Learning often matches or
#   beats DQN. The Q-table can represent every state exactly; the neural net
#   must approximate. DQN's advantage appears at scale — thousands of states,
#   continuous inputs, visual observations — where a table becomes infeasible.
#   If DQN wins here, great. If tabular wins, that's the expected result and
#   the interesting lesson: complexity ≠ better on small problems.
#
# CONCEPT — Greedy evaluation
#   Training uses epsilon-greedy (random exploration). Evaluation uses epsilon=0
#   (pure greedy — always take the best known action). This separates learning
#   from performance: we want to know how good the LEARNED policy is, not how
#   well the agent explores.
#
# Self-driving analog:
#   Training = practice drives with occasional random turns to explore.
#   Evaluation = the actual trip with no randomness — does the agent know the route?

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import torch
import pathplanning
from collections import defaultdict
from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer

# ---- Shared hyperparameters --------------------------------------------------

GRID_WIDTH    = 41
GRID_HEIGHT   = 41
GOAL_X        = 38
GOAL_Y        = 40
OBSTACLE_DENSITY = 0.3
MAZE_SEED     = 42          # fixed seed: both agents train on identical maze

TRAIN_EPISODES   = 1000
EVAL_EPISODES    = 100
MAX_STEPS        = GRID_WIDTH * GRID_HEIGHT * 4

GAMMA            = 0.95
LEARNING_RATE    = 0.1      # tabular
DQN_LR           = 0.001
EPSILON_START    = 1.0
EPSILON_MIN      = 0.05
EPSILON_DECAY    = 0.995

# ---- State helper ------------------------------------------------------------

def state_to_tensor(state: list, env) -> torch.Tensor:
    normalised = [
        state[0] / GRID_WIDTH,
        state[1] / GRID_HEIGHT,
    ] + list(env.getLineOfSight(state[0], state[1]))
    return torch.FloatTensor(normalised).unsqueeze(0)

# ---- Tabular Q-Learning ------------------------------------------------------
# CONCEPT — Q-table as a dictionary
#   For grid RL the Q-table maps (x, y, action) → expected return.
#   A defaultdict(float) returns 0.0 for any unseen state-action pair, which is
#   the correct uninformed prior (neither optimistic nor pessimistic).

class TabularQLearner:

    def __init__(self):
        self.q_table = defaultdict(float)
        self.epsilon = EPSILON_START

    def _key(self, state: list, action: int) -> tuple:
        return (int(state[0]), int(state[1]), action)

    def get_q(self, state: list, action: int) -> float:
        return self.q_table[self._key(state, action)]

    def best_action(self, state: list) -> int:
        q_values = [self.get_q(state, action) for action in range(4)]
        return int(max(range(4), key=lambda action: q_values[action]))

    def select_action(self, state: list) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return self.best_action(state)

    def update(self, state: list, action: int, reward: float,
               next_state: list, done: bool) -> None:
        # CONCEPT — Bellman update
        #   Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        #   The bracketed term is the TD error: how wrong was the current estimate?
        #   Positive TD error → Q was too low → increase it.
        #   Negative TD error → Q was too high → decrease it.
        best_next = 0.0 if done else max(self.get_q(next_state, a) for a in range(4))
        td_target = reward + GAMMA * best_next
        td_error  = td_target - self.get_q(state, action)
        self.q_table[self._key(state, action)] += LEARNING_RATE * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


def train_tabular(env) -> TabularQLearner:
    agent = TabularQLearner()
    for episode in range(1, TRAIN_EPISODES + 1):
        state = list(env.reset())
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            result = env.step(action)
            next_state = [int(result[0]), int(result[1])]
            reward     = float(result[2])
            done       = bool(result[3])
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()
        if episode % 200 == 0:
            print(f"  [Q-Learning] Episode {episode}/{TRAIN_EPISODES} | "
                  f"epsilon: {agent.epsilon:.3f} | "
                  f"Q-table entries: {len(agent.q_table)}")
    return agent


# ---- DQN ---------------------------------------------------------------------

def train_dqn(env) -> DQNNetwork:
    STATE_SIZE   = 6
    ACTION_SIZE  = 4
    HIDDEN_SIZE  = 128
    BATCH_SIZE   = 64
    BUFFER_CAP   = 10000
    TARGET_FREQ  = 100

    main_network   = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_network = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_network.load_state_dict(main_network.state_dict())
    optimizer      = torch.optim.Adam(main_network.parameters(), lr=DQN_LR)
    buffer         = ReplayBuffer(BUFFER_CAP)
    epsilon        = EPSILON_START
    total_steps    = 0

    for episode in range(1, TRAIN_EPISODES + 1):
        state        = list(env.reset())
        state_tensor = state_to_tensor(state, env)

        for _ in range(MAX_STEPS):
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    action = main_network(state_tensor).argmax().item()

            result           = env.step(action)
            next_state       = [int(result[0]), int(result[1])]
            reward           = float(result[2])
            done             = bool(result[3])
            next_state_tensor = state_to_tensor(next_state, env)

            buffer.push(state_tensor, action, reward, next_state_tensor, done)
            state_tensor = next_state_tensor
            total_steps += 1

            if buffer.is_ready(BATCH_SIZE):
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
                q_pred = main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # Double DQN: main selects, target evaluates.
                    selected_actions   = main_network(next_states).argmax(1)
                    next_q            = target_network(next_states).gather(
                        1, selected_actions.unsqueeze(1)
                    ).squeeze(1)
                q_target = rewards + GAMMA * next_q * (1 - dones)
                loss = torch.nn.MSELoss()(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % TARGET_FREQ == 0:
                target_network.load_state_dict(main_network.state_dict())

            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if episode % 200 == 0:
            print(f"  [DQN]        Episode {episode}/{TRAIN_EPISODES} | "
                  f"epsilon: {epsilon:.3f} | "
                  f"buffer: {len(buffer)}")

    return main_network


# ---- Evaluation --------------------------------------------------------------

def evaluate_tabular(agent: TabularQLearner, env) -> dict:
    successes   = 0
    total_steps = 0
    for _ in range(EVAL_EPISODES):
        state = list(env.reset())
        for step in range(MAX_STEPS):
            action     = agent.best_action(state)
            result     = env.step(action)
            state      = [int(result[0]), int(result[1])]
            done       = bool(result[3])
            if done:
                successes   += 1
                total_steps += step + 1
                break
    success_rate = successes / EVAL_EPISODES
    avg_steps    = total_steps / max(successes, 1)
    return {"success_rate": success_rate, "avg_steps_on_success": avg_steps,
            "successes": successes}


def evaluate_dqn(network: DQNNetwork, env) -> dict:
    successes   = 0
    total_steps = 0
    for _ in range(EVAL_EPISODES):
        state        = list(env.reset())
        state_tensor = state_to_tensor(state, env)
        for step in range(MAX_STEPS):
            with torch.no_grad():
                action = network(state_tensor).argmax().item()
            result           = env.step(action)
            state            = [int(result[0]), int(result[1])]
            done             = bool(result[3])
            state_tensor     = state_to_tensor(state, env)
            if done:
                successes   += 1
                total_steps += step + 1
                break
    success_rate = successes / EVAL_EPISODES
    avg_steps    = total_steps / max(successes, 1)
    return {"success_rate": success_rate, "avg_steps_on_success": avg_steps,
            "successes": successes}


# ---- Entry point -------------------------------------------------------------

def compare():
    print("\n" + "=" * 60)
    print("Algorithm Comparison: Tabular Q-Learning vs DQN")
    print("=" * 60)
    print(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT} | Episodes: {TRAIN_EPISODES} each")
    print(f"Same maze, same goal, same episode budget.")
    print(f"Evaluation: {EVAL_EPISODES} greedy episodes (epsilon=0).\n")

    env = pathplanning.GridEnvironment(
        GRID_WIDTH, GRID_HEIGHT, 0, 0, GOAL_X, GOAL_Y, OBSTACLE_DENSITY, MAZE_SEED
    )

    print("Training Tabular Q-Learning...")
    tabular_agent = train_tabular(env)

    print("\nTraining DQN...")
    dqn_network = train_dqn(env)

    print("\nEvaluating both policies on the same maze...")
    tabular_results = evaluate_tabular(tabular_agent, env)
    dqn_results     = evaluate_dqn(dqn_network, env)

    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Q-Learning':>15} {'DQN':>15}")
    print("-" * 60)
    print(f"{'Success rate':<30} "
          f"{tabular_results['success_rate']*100:>14.1f}% "
          f"{dqn_results['success_rate']*100:>14.1f}%")
    print(f"{'Episodes succeeded':<30} "
          f"{tabular_results['successes']:>15d} "
          f"{dqn_results['successes']:>15d}")
    print(f"{'Avg steps (successes only)':<30} "
          f"{tabular_results['avg_steps_on_success']:>15.1f} "
          f"{dqn_results['avg_steps_on_success']:>15.1f}")
    print(f"{'Q-table / network params':<30} "
          f"{len(tabular_agent.q_table):>15,} "
          f"{'~41k':>15}")
    print("=" * 60)
    print("\nInterpretation:")
    print("  If Q-Learning wins: small grids favour exact lookup over approximation.")
    print("  If DQN wins: the neural net generalised better with this obstacle density.")
    print("  Neither result is wrong — both reveal something real about the algorithms.")


if __name__ == "__main__":
    compare()
