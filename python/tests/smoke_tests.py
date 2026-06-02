# python/tests/smoke_tests.py
# Fast regression checks for every Python network, agent, and buffer class.
# No training, no environment — pure construction + forward pass + shape checks.
# Runs in under 5 seconds. Run after any change to networks/ or agents/.
#
# Mirrors the style of tests/SmokeTests.cpp:
#   [PASS] / [FAIL] per check, summary at end, non-zero exit on any failure.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from torch.distributions import Categorical

# Networks
from dqn_network                import DQNNetwork
from networks.actor_critic      import ActorCriticNetwork
from networks.sac_network       import SACActor, SACCritic
from networks.lstm_policy       import LSTMActorCriticNetwork
from networks.alphazero_network import AlphaZeroNetwork
from networks.world_model       import DynamicsNetwork, RewardNetwork, PolicyNetwork, action_to_onehot

# Agents
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

# Buffers
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# ---- Test harness ------------------------------------------------------------

_passed = 0
_failed = 0


def check(condition: bool, test_name: str) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {test_name}")
    else:
        _failed += 1
        print(f"  [FAIL] {test_name}")


def check_near(actual: float, expected: float, tolerance: float, test_name: str) -> None:
    check(abs(actual - expected) <= tolerance, test_name)


# ---- Shared fixture ----------------------------------------------------------

_DUMMY_STATE = torch.FloatTensor([[0.5, 0.3, 0.0, 1.0, 0.0, 1.0]])  # [1, 6]
_ACTION_ONEHOT = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0]])            # action=0


# ---- DQN network -------------------------------------------------------------

def test_dqn_network():
    print("\n-- DQNNetwork (Dueling) --")
    network = DQNNetwork(state_size=6, action_size=4, hidden_size=128)

    output = network(_DUMMY_STATE)
    check(output.shape == torch.Size([1, 4]),
          "DQNNetwork forward output shape [1, 4]")

    # Dueling: V(s) + A(s,a) - mean(A). Q-values should be real numbers, not all equal.
    check(not torch.isnan(output).any().item(),
          "DQNNetwork output contains no NaN")

    batch = _DUMMY_STATE.repeat(8, 1)
    batch_output = network(batch)
    check(batch_output.shape == torch.Size([8, 4]),
          "DQNNetwork batch forward shape [8, 4]")


# ---- SAC networks ------------------------------------------------------------

def test_sac_networks():
    print("\n-- SACActor + SACCritic --")
    actor  = SACActor(state_size=6, action_size=4, hidden_size=128)
    critic = SACCritic(state_size=6, action_size=4, hidden_size=128)

    logits = actor(_DUMMY_STATE)
    check(logits.shape == torch.Size([1, 4]),
          "SACActor forward logits shape [1, 4]")

    action_index, log_prob, entropy = actor.get_action_and_log_prob(_DUMMY_STATE)
    check(isinstance(action_index, int) and 0 <= action_index <= 3,
          "SACActor sampled action in [0, 3]")
    check(not torch.isnan(log_prob).any().item(),
          "SACActor log_prob is finite")
    check(entropy.item() > 0.0,
          "SACActor entropy positive (uniform init → all actions roughly equal)")

    q_values = critic(_DUMMY_STATE)
    check(q_values.shape == torch.Size([1, 4]),
          "SACCritic forward Q-values shape [1, 4]")
    check(not torch.isnan(q_values).any().item(),
          "SACCritic output contains no NaN")


# ---- Actor-Critic (PPO) ------------------------------------------------------

def test_actor_critic_network():
    print("\n-- ActorCriticNetwork (PPO) --")
    network = ActorCriticNetwork(state_size=6, action_size=4, hidden_size=128)

    action_probs, value = network(_DUMMY_STATE)
    check(action_probs.shape == torch.Size([1, 4]),
          "ActorCriticNetwork action_probs shape [1, 4]")
    check(value.shape == torch.Size([1, 1]),
          "ActorCriticNetwork value shape [1, 1]")
    check_near(action_probs.sum().item(), 1.0, 1e-5,
               "ActorCriticNetwork action_probs sum to 1.0")
    check((action_probs >= 0).all().item(),
          "ActorCriticNetwork action_probs all non-negative")


# ---- LSTM Actor-Critic -------------------------------------------------------

def test_lstm_actor_critic():
    print("\n-- LSTMActorCriticNetwork --")
    network = LSTMActorCriticNetwork(state_size=6, action_size=4, hidden_size=128)

    hidden_state = network.get_initial_hidden_state()
    check(hidden_state[0].shape == torch.Size([1, 1, 128]),
          "LSTM initial h_0 shape [1, 1, 128]")
    check(hidden_state[1].shape == torch.Size([1, 1, 128]),
          "LSTM initial c_0 shape [1, 1, 128]")

    action_probs, value, new_hidden = network(_DUMMY_STATE, hidden_state)
    check(action_probs.shape == torch.Size([1, 4]),
          "LSTMActorCritic action_probs shape [1, 4]")
    check(value.shape == torch.Size([1, 1]),
          "LSTMActorCritic value shape [1, 1]")
    check_near(action_probs.sum().item(), 1.0, 1e-5,
               "LSTMActorCritic action_probs sum to 1.0")
    check(new_hidden[0].shape == hidden_state[0].shape,
          "LSTMActorCritic hidden state shape preserved after step")

    # Two sequential steps — hidden state carries memory between them.
    _, _, hidden_step2 = network(_DUMMY_STATE, new_hidden)
    check(hidden_step2[0].shape == hidden_state[0].shape,
          "LSTMActorCritic hidden state stable across two steps")


# ---- AlphaZero network -------------------------------------------------------

def test_alphazero_network():
    print("\n-- AlphaZeroNetwork --")
    network = AlphaZeroNetwork(state_size=6, action_size=4, hidden_size=128)

    policy_probs, value = network(_DUMMY_STATE)
    check(policy_probs.shape == torch.Size([1, 4]),
          "AlphaZeroNetwork policy_probs shape [1, 4]")
    check(value.shape == torch.Size([1, 1]),
          "AlphaZeroNetwork value shape [1, 1]")
    check_near(policy_probs.sum().item(), 1.0, 1e-5,
               "AlphaZeroNetwork policy_probs sum to 1.0")

    # CONCEPT — tanh value output
    #   AlphaZero value head uses tanh: output must be in (-1, 1).
    #   +1 = guaranteed win (goal reached), -1 = guaranteed loss (timeout).
    value_scalar = value.item()
    check(-1.0 <= value_scalar <= 1.0,
          f"AlphaZeroNetwork value in [-1, 1] via tanh (got {value_scalar:.4f})")


# ---- World model networks ----------------------------------------------------

def test_world_model_networks():
    print("\n-- World Model: DynamicsNetwork / RewardNetwork / PolicyNetwork --")

    dynamics_net = DynamicsNetwork(state_size=6, action_size=4, hidden_size=128)
    reward_net   = RewardNetwork(state_size=6,   action_size=4, hidden_size=64)
    policy_net   = PolicyNetwork(state_size=6,   action_size=4, hidden_size=128)

    predicted_position = dynamics_net(_DUMMY_STATE, _ACTION_ONEHOT)
    check(predicted_position.shape == torch.Size([1, 2]),
          "DynamicsNetwork output shape [1, 2] (position only, not full state)")
    check(not torch.isnan(predicted_position).any().item(),
          "DynamicsNetwork output contains no NaN")

    predicted_reward = reward_net(_DUMMY_STATE, _ACTION_ONEHOT)
    check(predicted_reward.shape == torch.Size([1, 1]),
          "RewardNetwork output shape [1, 1]")

    action_logits = policy_net(_DUMMY_STATE)
    check(action_logits.shape == torch.Size([1, 4]),
          "PolicyNetwork output shape [1, 4] (raw logits, no softmax)")

    onehot = action_to_onehot(2, action_size=4)
    check(onehot.shape == torch.Size([1, 4]),
          "action_to_onehot shape [1, 4]")
    check(onehot[0, 2].item() == 1.0 and onehot[0, 0].item() == 0.0,
          "action_to_onehot correct one-hot for action=2")

    # Batch forward pass.
    batch_state  = _DUMMY_STATE.repeat(4, 1)
    batch_onehot = _ACTION_ONEHOT.repeat(4, 1)
    batch_pos = dynamics_net(batch_state, batch_onehot)
    check(batch_pos.shape == torch.Size([4, 2]),
          "DynamicsNetwork batch shape [4, 2]")


# ---- Replay buffers ----------------------------------------------------------

def test_replay_buffer():
    print("\n-- ReplayBuffer --")
    buffer = ReplayBuffer(capacity=100)

    check(len(buffer) == 0,           "ReplayBuffer empty on construction")
    check(not buffer.is_ready(1),     "ReplayBuffer not ready before any push")

    buffer.push(_DUMMY_STATE, 0, -1.0, _DUMMY_STATE, False)
    check(len(buffer) == 1,           "ReplayBuffer len=1 after one push")
    check(not buffer.is_ready(10),    "ReplayBuffer not ready for batch=10 with 1 sample")

    for _ in range(63):
        buffer.push(_DUMMY_STATE, 1, -1.0, _DUMMY_STATE, False)

    check(len(buffer) == 64,          "ReplayBuffer len=64 after 64 pushes")
    check(buffer.is_ready(64),        "ReplayBuffer ready for batch=64")

    states, actions, rewards, next_states, dones = buffer.sample(32)
    check(states.shape      == torch.Size([32, 6]), "ReplayBuffer sample states shape [32, 6]")
    check(actions.shape     == torch.Size([32]),    "ReplayBuffer sample actions shape [32]")
    check(rewards.shape     == torch.Size([32]),    "ReplayBuffer sample rewards shape [32]")
    check(dones.dtype       == torch.float32,       "ReplayBuffer sample dones dtype float32")

    # Circular overwrite: push 50 more into capacity=100 buffer already holding 64.
    for _ in range(50):
        buffer.push(_DUMMY_STATE, 2, 0.0, _DUMMY_STATE, True)
    check(len(buffer) == 100, "ReplayBuffer caps at capacity after overflow")


def test_prioritized_replay_buffer():
    print("\n-- PrioritizedReplayBuffer --")
    per_buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

    for i in range(20):
        per_buffer.push(_DUMMY_STATE, i % 4, float(i), _DUMMY_STATE, False)

    check(len(per_buffer) == 20, "PrioritizedReplayBuffer len=20 after 20 pushes")
    check(per_buffer.is_ready(10), "PrioritizedReplayBuffer ready for batch=10")

    result = per_buffer.sample(batch_size=8, beta=0.4)
    check(len(result) == 7,
          "PrioritizedReplayBuffer sample returns 7 elements (states,actions,rewards,next,dones,indices,weights)")

    states, actions, rewards, next_states, dones, indices, weights = result
    check(states.shape   == torch.Size([8, 6]), "PER sample states shape [8, 6]")
    check(weights.shape  == torch.Size([8]),    "PER importance weights shape [8]")
    check((weights > 0).all().item(),           "PER importance weights all positive")

    per_buffer.update_priorities(indices, [abs(r) + 1e-5 for r in rewards.tolist()])
    check(True, "PrioritizedReplayBuffer update_priorities runs without error")


# ---- Agents ------------------------------------------------------------------

def test_ppo_agent():
    print("\n-- PPOAgent --")
    network = ActorCriticNetwork(state_size=6, action_size=4)
    agent   = PPOAgent(network=network, learning_rate=3e-4, gamma=0.95, epochs=4)

    action, log_prob, value = agent.select_action(_DUMMY_STATE)
    check(0 <= action <= 3,                     "PPOAgent select_action in [0, 3]")
    check(not torch.isnan(log_prob).item(),      "PPOAgent log_prob is finite")
    check(value.shape == torch.Size([1, 1]),     "PPOAgent value shape [1, 1]")

    rewards = [-1.0, -1.0, -1.0, 100.0]
    dones   = [False, False, False, True]
    returns = agent.compute_returns(rewards, dones)
    check(returns.shape == torch.Size([4]),      "PPOAgent compute_returns shape [4]")
    check(returns[-1].item() > returns[0].item(),
          "PPOAgent returns: later steps closer to reward have higher return")


def test_sac_agent():
    print("\n-- SACAgent --")
    actor      = SACActor(state_size=6, action_size=4)
    critic_one = SACCritic(state_size=6, action_size=4)
    critic_two = SACCritic(state_size=6, action_size=4)
    target_entropy = math.log(4)  # entropy of uniform over 4 actions

    agent = SACAgent(actor, critic_one, critic_two,
                     learning_rate=3e-4, gamma=0.95, tau=0.005,
                     action_size=4)

    action = agent.select_action(_DUMMY_STATE)
    check(0 <= action <= 3, "SACAgent select_action returns valid action in [0, 3]")

    # Run one update with a minimal fake batch.
    fake_states      = _DUMMY_STATE.repeat(8, 1)
    fake_actions     = torch.LongTensor([0, 1, 2, 3, 0, 1, 2, 3])
    fake_rewards     = torch.FloatTensor([-1.0] * 8)
    fake_next_states = _DUMMY_STATE.repeat(8, 1)
    fake_dones       = torch.zeros(8)

    fake_batch = (fake_states, fake_actions, fake_rewards, fake_next_states, fake_dones)
    critic_loss, actor_loss, alpha_loss = agent.update(fake_batch)

    check(isinstance(critic_loss, float) and not math.isnan(critic_loss),
          "SACAgent update critic_loss is finite float")
    check(isinstance(actor_loss,  float) and not math.isnan(actor_loss),
          "SACAgent update actor_loss is finite float")
    check(isinstance(alpha_loss,  float) and not math.isnan(alpha_loss),
          "SACAgent update alpha_loss is finite float")


# ---- Entry point -------------------------------------------------------------

def run_all() -> int:
    global _passed, _failed
    _passed = 0
    _failed = 0

    print("\n============== Python Smoke Tests ==============")
    print("Shape checks, forward passes, agent construction.")
    print("No training, no environment — runs in under 5 seconds.\n")

    test_dqn_network()
    test_sac_networks()
    test_actor_critic_network()
    test_lstm_actor_critic()
    test_alphazero_network()
    test_world_model_networks()
    test_replay_buffer()
    test_prioritized_replay_buffer()
    test_ppo_agent()
    test_sac_agent()

    print(f"\nResults: {_passed} passed, {_failed} failed.")

    if _failed > 0:
        print("SMOKE TEST FAILURE — fix before committing.")
        return 1
    print("All Python smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
