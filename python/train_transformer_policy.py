# python/train_transformer_policy.py
#
# PURPOSE: PPO training with a Transformer-based actor-critic replacing the MLP.
#          The training loop is identical to train_ppo.py — only the network changes.
#
# CORE CONCEPT — What changes vs train_ppo.py
#   train_ppo.py:            state = [x/w, y/h, wall_up, wall_down, wall_left, wall_right]
#                            network = ActorCriticNetwork (MLP, 6 → 128 → 128 → 4+1)
#   train_transformer_policy.py: state = GridEncoder.build_grid_tensor(env, x, y)
#                            network = TransformerActorCritic (patch embed → attention → 4+1)
#
#   The PPO agent, advantage computation, policy loss, and value loss are UNCHANGED.
#   This is the whole point: a modular architecture where you can swap the backbone.
#
# CORE CONCEPT — Why the richer state matters
#   MLP with 6 features: the agent knows it is near a wall, nothing else.
#   Transformer with full grid: the agent sees where all walls are, where the goal is,
#   and can attend to distant patches to build a mental map.
#   Expected result: faster convergence on complex mazes and higher final success rate.
#
# CORE CONCEPT — Computational cost tradeoff
#   MLP forward pass: ~10,000 multiply-adds.
#   Transformer forward (grid encoder + 2 attention layers on ~100 tokens): ~4M multiply-adds.
#   This is 400× more compute per step. Why might it still be worth it?
#   Transformer may converge in 5× fewer episodes → net 80× faster wall-clock training.
#   Or it achieves 20% higher success rate on hard mazes, which no MLP can match.
#   The benchmark at the end of Phase 10 measures whether the tradeoff pays off here.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pathplanning
from networks.transformer_policy import TransformerActorCritic
from networks.grid_encoder       import GridEncoder
from agents.ppo_agent            import PPOAgent

# ---- Hyperparameters --------------------------------------------------------

GRID_HEIGHT   = 41
GRID_WIDTH    = 41
NUM_ACTIONS   = 4

EPISODES      = 600
MAX_STEPS     = 300   # optimal path ≤ ~80 steps; 300 gives headroom without millions of wasted steps

LEARNING_RATE = 0.0001   # lower than MLP PPO — Transformer is sensitive to large steps
GAMMA         = 0.95
PPO_EPOCHS    = 4

SAVE_PATH     = os.path.join(os.path.dirname(__file__), 'data', 'transformer_policy.pt')


def build_state_tensor(
    environment:  pathplanning.GridEnvironment,
    agent_x:      int,
    agent_y:      int
) -> torch.Tensor:
    """Full build — used at benchmark time. Training uses the cached path instead."""
    return GridEncoder.build_grid_tensor(environment, agent_x, agent_y, GRID_HEIGHT, GRID_WIDTH)


def _build_static_channels(environment: pathplanning.GridEnvironment) -> torch.Tensor:
    """
    Build channels 0 (obstacles) and 2 (goal) once — both are constant for the
    lifetime of one GridEnvironment. Channel 1 (agent) is left as zeros.

    The training loop updates only channel 1 in-place each step (2 tensor
    writes, zero FFI calls) instead of rebuilding all 41×41=1681 cells every step.
    """
    grid = torch.zeros(1, 3, GRID_HEIGHT, GRID_WIDTH, dtype=torch.float32)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if environment.isObstacle(x, y):
                grid[0, 0, y, x] = 1.0
    goal = environment.getGoal()
    grid[0, 2, goal[1], goal[0]] = 1.0
    return grid


def train() -> None:
    """
    CONCEPT — Identical PPO loop, different state/network:
    Compare carefully with train_ppo.py. The loop structure is IDENTICAL.
    The differences are exactly two lines:
      1. build_state_tensor() instead of state_to_tensor()
      2. TransformerActorCritic instead of ActorCriticNetwork

    This demonstrates modular design: the PPO algorithm is independent of the
    network architecture. Swap the network, keep the algorithm. This is why
    separate files (actor_critic.py vs transformer_policy.py) exist.

    Implement:
    1. Create GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1, GRID_WIDTH-2, GRID_HEIGHT-2, 0.3)
    2. Instantiate TransformerActorCritic(GRID_HEIGHT, GRID_WIDTH)
    3. Instantiate PPOAgent(network, LEARNING_RATE, GAMMA, PPO_EPOCHS)
       (PPOAgent from agents/ppo_agent.py — reused without modification)

    4. Episode loop (identical structure to train_ppo.py):
       a. env.reset() → [x, y]
       b. build_state_tensor(env, x, y) → state_tensor
       c. For each step:
            agent.select_action(state_tensor) → action, log_prob, value
            env.step(action) → next_state, reward, done
            collect (state, action, log_prob, reward, done, value)
       d. Compute returns: agent.compute_returns(rewards, dones)
       e. Compute advantages: returns - values (normalise to zero mean, unit std)
       f. agent.update(states, actions, log_probs, returns, advantages) → loss

    5. Print every 10 episodes: episode, reward, loss

    6. After training: torch.save(network.state_dict(), SAVE_PATH)
       Print: "Model saved to {SAVE_PATH}"
    """
    env     = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1, GRID_WIDTH - 2, GRID_HEIGHT - 2, 0.3)
    network = TransformerActorCritic(GRID_HEIGHT, GRID_WIDTH)
    agent   = PPOAgent(network, LEARNING_RATE, GAMMA, PPO_EPOCHS)

    # Build obstacle + goal channels ONCE — maze is fixed for this env instance.
    # Per-step cost drops from 1681 FFI calls to 2 tensor writes.
    static_grid = _build_static_channels(env)

    for episode in range(1, EPISODES + 1):

        states_list    = []
        actions_list   = []
        log_probs_list = []
        rewards_list   = []
        dones_list     = []
        values_list    = []

        state              = env.reset()
        curr_x, curr_y     = int(state[0]), int(state[1])

        # Reset agent channel and set initial position.
        static_grid[0, 1, :, :] = 0.0
        static_grid[0, 1, curr_y, curr_x] = 1.0
        state_tensor   = static_grid.clone()
        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            action, log_prob, value = agent.select_action(state_tensor)

            result              = env.step(action)
            next_x, next_y      = int(result[0]), int(result[1])
            reward              = result[2]
            done                = result[3] > 0.5

            states_list.append(state_tensor)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            dones_list.append(done)
            values_list.append(value)

            episode_reward += reward

            # Update agent channel in-place, clone for next state.
            static_grid[0, 1, curr_y, curr_x] = 0.0
            static_grid[0, 1, next_y, next_x] = 1.0
            state_tensor = static_grid.clone()
            curr_x, curr_y = next_x, next_y

            if done:
                break

        if len(rewards_list) == 0:
            continue

        returns       = agent.compute_returns(rewards_list, dones_list)
        values_tensor = torch.cat(values_list).squeeze()
        advantages    = returns - values_tensor.detach()
        advantages    = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_tensor    = torch.cat(states_list)
        actions_tensor   = torch.LongTensor(actions_list)
        log_probs_tensor = torch.stack(log_probs_list).detach()

        loss = agent.update(states_tensor, actions_tensor, log_probs_tensor, returns, advantages)

        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Reward: {episode_reward:8.1f} | Loss: {loss:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(network.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == '__main__':
    train()
