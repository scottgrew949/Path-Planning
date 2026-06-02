# python/train_lstm_ppo.py
# PPO training loop with a recurrent LSTM policy for partial-observability navigation.
#
# CONCEPT — Why not reuse PPOAgent from ppo_agent.py?
#   The standard PPOAgent.update() batches all time-steps independently:
#     states_tensor shape = [T, state_size]
#   It then runs one forward pass over the whole batch — no sequence structure.
#   For an LSTM the sequence ORDER matters: step 5's hidden state depends on
#   steps 0-4.  We must feed the episode as a TIME SEQUENCE [1, T, hidden_size]
#   so that BPTT gradients can flow back through every hidden state.
#   That requires restructuring both the forward pass and the update, so the
#   cleanest design is to write the PPO logic inline here rather than bolt it
#   onto the existing stateless agent class.
#
# CONCEPT — On-policy training (reminder)
#   PPO is on-policy: each episode's data is collected with the CURRENT policy,
#   used for PPO_EPOCHS gradient steps, then discarded.  The recurrent hidden
#   states that were produced during collection are stored so we can re-run the
#   LSTM over the same sequence during the update and get the same feature vectors
#   (modulo the gradient being live).
#
# CONCEPT — Generalised Advantage Estimation (GAE)
#   Simple advantage: A_t = G_t - V(s_t)  (return minus baseline).
#   Problem: G_t has high variance (random future rewards).
#   GAE blends one-step TD errors via exponential decay:
#     δ_t = r_t + γ·V(s_{t+1}) - V(s_t)          (TD residual)
#     A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
#   λ=1  → Monte-Carlo returns (unbiased, high variance)
#   λ=0  → one-step TD (biased, low variance)
#   λ=0.95 → the empirically best sweet-spot used in the original PPO paper.
#
# CONCEPT — BPTT in the PPO update
#   During ppo_update we feed the full episode sequence [1, T, hidden_size] to
#   the LSTM in a single forward pass.  PyTorch's autograd unrolls the computation
#   graph across all T steps, so loss.backward() computes gradients w.r.t. every
#   weight that touched every hidden state — this is Backpropagation Through Time.
#   The starting hidden state for the update sequence is h_0 = zeros, matching
#   what was used at the beginning of the episode during rollout.
#
# Self-driving analog:
#   The LSTM hidden state is like the driver's short-term working memory —
#   "I just came from that junction, the left lane was blocked."  Without it,
#   the agent in a fog-of-war maze has no way to avoid re-entering dead ends
#   it already visited.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.distributions import Categorical
import pathplanning
from networks.lstm_policy import LSTMActorCriticNetwork

# ---- Hyperparameters ---------------------------------------------------------

GRID_WIDTH             = 41
GRID_HEIGHT            = 41
STATE_SIZE             = 6        # [x/w, y/h, wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE            = 4        # 0=UP 1=DOWN 2=LEFT 3=RIGHT
HIDDEN_SIZE            = 128

EPISODES               = 2000
MAX_STEPS              = GRID_WIDTH * GRID_HEIGHT * 4

LEARNING_RATE          = 0.0003
GAMMA                  = 0.95
GAE_LAMBDA             = 0.95     # λ for Generalised Advantage Estimation
PPO_EPOCHS             = 4
PPO_CLIP               = 0.2      # ε for clipped surrogate objective
ENTROPY_COEFFICIENT    = 0.01     # entropy bonus weight — keeps policy from collapsing
VALUE_LOSS_COEFFICIENT = 0.5      # critic loss weight relative to actor loss

# ---- State encoding ----------------------------------------------------------

def state_to_tensor(state: list, width: int, height: int, env=None) -> torch.Tensor:
    """Encode raw [x, y] state into a normalised 6-dim feature tensor.

    Normalising x and y to [0, 1] prevents the network from learning that
    absolute pixel coordinates matter more than wall proximity.  Wall flags
    are already binary so no normalisation needed.
    """
    normalised_features = [state[0] / width, state[1] / height]
    if env is not None:
        normalised_features += env.getLineOfSight(state[0], state[1])
    else:
        normalised_features += [0, 0, 0, 0]
    return torch.FloatTensor(normalised_features).unsqueeze(0)  # [1, STATE_SIZE]

# ---- GAE computation ---------------------------------------------------------

def compute_gae(
    rewards:      list,
    values:       list,
    dones:        list,
    gamma:        float,
    gae_lambda:   float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalised Advantage Estimates and discounted returns.

    Parameters
    ----------
    rewards    : list of float, length T
    values     : list of torch.Tensor shape [1,1], length T  (critic estimates)
    dones      : list of bool, length T
    gamma      : discount factor
    gae_lambda : GAE λ decay

    Returns
    -------
    advantages : FloatTensor shape [T]  — normalised, zero-mean
    returns    : FloatTensor shape [T]  — advantages + detached values (targets for critic)
    """
    # CONCEPT — Bootstrapping at episode boundaries
    #   When done=True at step t, the episode ended — there is no future reward.
    #   We must zero the bootstrap value V(s_{t+1}) so the TD residual becomes
    #   simply r_t - V(s_t).  The (1 - done_mask) term implements this.

    episode_length = len(rewards)
    values_flat    = [value_tensor.item() for value_tensor in values]

    advantages_reversed = []
    running_gae_sum     = 0.0

    for time_index in reversed(range(episode_length)):
        # Bootstrap: next value is 0 if episode terminated at this step.
        done_mask        = 1.0 - float(dones[time_index])
        next_value       = values_flat[time_index + 1] if time_index + 1 < episode_length else 0.0
        next_value       = next_value * done_mask

        td_residual      = rewards[time_index] + gamma * next_value - values_flat[time_index]

        # GAE accumulates backwards: A_t = δ_t + γλ·A_{t+1}
        running_gae_sum  = td_residual + gamma * gae_lambda * done_mask * running_gae_sum
        advantages_reversed.append(running_gae_sum)

    advantages_reversed.reverse()
    advantages = torch.FloatTensor(advantages_reversed)

    # Normalise advantages — zero mean, unit variance.
    # Keeps gradient magnitudes stable across episodes of different lengths.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    values_tensor = torch.FloatTensor(values_flat)
    returns       = advantages + values_tensor  # used as critic regression target

    return advantages, returns

# ---- PPO update --------------------------------------------------------------

def ppo_update(
    network:             LSTMActorCriticNetwork,
    optimizer:           torch.optim.Optimizer,
    episode_states:      list,    # list of T tensors, each [1, STATE_SIZE]
    episode_actions:     list,    # list of T ints
    old_log_probs:       torch.Tensor,  # [T]
    returns:             torch.Tensor,  # [T]  — critic regression targets
    advantages:          torch.Tensor,  # [T]  — normalised GAE
) -> float:
    """Run PPO_EPOCHS gradient steps on the collected episode.

    LSTM BPTT strategy
    ------------------
    We stack the full episode into one sequence [1, T, STATE_SIZE], run
    a single forward pass through the LSTM (starting from h_0=zeros, matching
    rollout), and collect action_probs and values for every time-step.
    PyTorch builds a computation graph that spans all T steps, so
    loss.backward() computes correct gradients through time.

    Why not re-use stored hidden states?
    -------------------------------------
    An alternative is to store the hidden state at each step during rollout and
    replay them.  That is fine for PPO_EPOCHS=1 but diverges on later epochs
    because the weights have changed — the stored hidden states were computed
    with OLD weights.  Starting from h_0=zeros every epoch is a minor
    approximation that is standard in recurrent PPO implementations (e.g.
    OpenAI's original code and CleanRL).
    """
    # Stack all per-step state tensors into a single sequence tensor.
    # Each element is [1, STATE_SIZE]; after stacking along dim=1 → [1, T, STATE_SIZE].
    episode_sequence = torch.cat(episode_states, dim=0).unsqueeze(0)  # [1, T, STATE_SIZE]
    actions_tensor   = torch.LongTensor(episode_actions)              # [T]

    total_loss_accumulated = 0.0

    for _ in range(PPO_EPOCHS):
        # Project entire sequence through input_projection + ReLU.
        projected_sequence = network.relu(network.input_projection(episode_sequence))
        # projected_sequence: [1, T, hidden_size]

        # Run LSTM over the full sequence in one shot — this is BPTT.
        # Hidden state starts at zeros (episode boundary reset).
        initial_hidden_state = network.get_initial_hidden_state(batch_size=1)
        lstm_output, _       = network.lstm(projected_sequence, initial_hidden_state)
        # lstm_output: [1, T, hidden_size]

        # Squeeze batch dim → [T, hidden_size] so heads operate per time-step.
        per_step_features = lstm_output.squeeze(0)  # [T, hidden_size]

        new_action_probs = torch.softmax(network.policy_head(per_step_features), dim=-1)
        new_values       = network.value_head(per_step_features).squeeze(-1)  # [T]

        distribution   = Categorical(new_action_probs)
        new_log_probs  = distribution.log_prob(actions_tensor)  # [T]
        entropy        = distribution.entropy().mean()

        # CONCEPT — Clipped surrogate objective
        #   ratio measures how much the policy has changed for each action taken.
        #   If ratio is too far from 1, the clip prevents the update from being
        #   too aggressive — conservative policy improvement.
        ratio           = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio   = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP)
        actor_loss      = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Critic: MSE between predicted values and GAE-based return targets.
        value_loss      = nn.functional.mse_loss(new_values, returns)

        # CONCEPT — Entropy bonus
        #   Maximising entropy keeps the policy spread over actions — prevents
        #   premature convergence to a deterministic policy before the maze is solved.
        #   Subtracted because we minimise the total loss.
        total_loss = (
            actor_loss
            + VALUE_LOSS_COEFFICIENT * value_loss
            - ENTROPY_COEFFICIENT   * entropy
        )

        optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping prevents exploding gradients through long LSTM sequences.
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss_accumulated += total_loss.item()

    return total_loss_accumulated / PPO_EPOCHS

# ---- Training loop -----------------------------------------------------------

def train():
    """Main LSTM-PPO training loop.

    Episode lifecycle
    -----------------
    1. Reset env → initial raw state.
    2. Reset LSTM hidden state to zeros (fresh episode memory).
    3. For each step:
       a. Encode state → tensor [1, STATE_SIZE].
       b. Forward pass → (action_probs, value, new_hidden_state).
       c. Sample action from Categorical(action_probs); record log_prob.
       d. Step env → reward, done.
       e. Store transition data.
    4. Compute GAE advantages and returns from stored episode.
    5. Run ppo_update on the collected episode sequence.
    """
    print("\n" + "=" * 60)
    print("LSTM PPO — Recurrent Policy for Partial Observability")
    print("=" * 60)
    print("Standard PPO (menu option 5) uses a flat MLP: one state snapshot → action.")
    print("Problem: with line-of-sight only, the same 4 wall flags can appear at")
    print("completely different maze positions — the task is non-Markovian.")
    print("A memoryless policy can't distinguish 'I already tried left and hit a wall'")
    print("from 'I haven't explored left yet.'")
    print("")
    print("LSTM fixes this: the hidden state (h_t, c_t) carries a compressed history")
    print("of the episode. The network remembers what it has seen and where it has been.")
    print("Hidden state resets to zeros at the start of each episode (fresh memory).")
    print("")
    print("PPO update: re-runs the full episode as one sequence through the LSTM each")
    print("epoch. Gradient clipping (max_norm=0.5) prevents exploding gradients through")
    print("long sequences.")
    print("")
    print("Self-driving analog: a car with no GPS knowing only 'wall left / wall right'.")
    print("The LSTM builds a mental map from the history of turns and observations.")
    print("")
    print("What to watch: total reward improving slower than flat PPO early on")
    print("(LSTM needs more episodes to learn sequence patterns), but eventually")
    print("outperforming it on long mazes where memory matters.")
    print("")
    print("  Episode      = one full run (or timeout at max steps)")
    print("  Reward       = total reward — negative early, rises as policy learns")
    print("  Loss         = combined actor + critic + entropy PPO loss")
    print("")

    env       = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 0, 0, 38, 40, 0.3)
    network   = LSTMActorCriticNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    for episode_number in range(1, EPISODES + 1):

        episode_states     = []
        episode_actions    = []
        episode_log_probs  = []
        episode_rewards    = []
        episode_dones      = []
        episode_values     = []

        raw_state    = env.reset()
        state_tensor = state_to_tensor(raw_state, GRID_WIDTH, GRID_HEIGHT, env)

        # Reset recurrent memory at the start of every episode.
        # The agent should not carry memory across independent episodes.
        hidden_state   = network.get_initial_hidden_state(batch_size=1)
        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            # Forward pass — returns probs, value estimate, updated hidden state.
            with torch.no_grad():
                action_probs, value_estimate, hidden_state = network.forward(
                    state_tensor, hidden_state
                )
                # Detach hidden state from the rollout graph — we rebuild it
                # from scratch during ppo_update using BPTT over the sequence.
                hidden_state = (
                    hidden_state[0].detach(),
                    hidden_state[1].detach(),
                )

            distribution = Categorical(action_probs)
            action_taken = distribution.sample()
            log_prob     = distribution.log_prob(action_taken)

            step_result = env.step(action_taken.item())
            next_state  = [int(step_result[0]), int(step_result[1])]
            reward      = float(step_result[2])
            done        = bool(step_result[3] > 0.5)

            episode_states.append(state_tensor)
            episode_actions.append(action_taken.item())
            episode_log_probs.append(log_prob.detach())
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_values.append(value_estimate.detach())

            episode_reward += reward
            state_tensor    = state_to_tensor(next_state, GRID_WIDTH, GRID_HEIGHT, env)

            if done:
                break

        if len(episode_rewards) == 0:
            continue

        advantages, returns = compute_gae(
            episode_rewards,
            episode_values,
            episode_dones,
            GAMMA,
            GAE_LAMBDA,
        )

        old_log_probs_tensor = torch.stack(episode_log_probs)  # [T]

        episode_loss = ppo_update(
            network,
            optimizer,
            episode_states,
            episode_actions,
            old_log_probs_tensor,
            returns,
            advantages,
        )

        if episode_number % 10 == 0 or episode_number <= 20:
            print(
                f"Episode {episode_number:4d} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Steps: {len(episode_rewards):5d} | "
                f"Loss: {episode_loss:.4f}"
            )


if __name__ == "__main__":
    train()
