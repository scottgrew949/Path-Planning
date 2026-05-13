# python/replay_buffer.py
# Experience replay buffer for DQN training.
#
# CONCEPT — What is experience replay?
#   During training the agent generates a stream of (state, action, reward,
#   next_state, done) tuples — one per step. Naively training on each tuple
#   the moment it's generated has two problems:
#     1. Consecutive steps are highly correlated — the net overfits to recent
#        experience and forgets earlier exploration.
#     2. The data distribution shifts as the policy improves — unstable training.
#
#   Experience replay fixes both: store every tuple in a buffer, then sample
#   random BATCHES for training. Random sampling breaks correlation.
#   The buffer holds old AND new experience — diverse, stable distribution.
#
# CONCEPT — Circular buffer
#   The buffer has a fixed max size. When full, new experiences overwrite the
#   oldest ones (circular/ring buffer). This keeps memory bounded while
#   naturally discarding stale experience from an early, worse policy.
#
# Self-driving analog:
#   The replay buffer is the car's dashcam footage archive. Training on random
#   clips from the last 10,000 miles is more stable than only training on
#   what happened in the last 10 seconds.

import random
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        pass

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        pass

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        #
        actions_tensor     = torch.LongTensor(actions)
        rewards_tensor     = torch.FloatTensor(rewards)
        dones_tensor       = torch.FloatTensor(dones)
        states_tensor      = torch.cat(states)
        next_states_tensor = torch.cat(next_states)
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
        
        # CONCEPT — Why FloatTensor vs LongTensor?
        #   Neural net inputs and rewards are floats — continuous values.
        #   Actions are indices (0-3) — integers used to index into Q-value output.
        #   PyTorch requires the correct dtype for each operation.
        pass

    def __len__(self) -> int:
        return len(self.buffer)
        pass

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size
        #       Training should not start until the buffer has enough samples
        #       to fill at least one batch — otherwise sample() would fail.
        pass
