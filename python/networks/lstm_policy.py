# python/networks/lstm_policy.py
# LSTM Actor-Critic network for recurrent PPO with partial observability.
#
# CONCEPT — Why LSTM for pathfinding?
#   In a fully-observed grid, every state_tensor encodes the complete picture.
#   With partial observability (line-of-sight only), a single snapshot is
#   ambiguous — the agent sees four wall flags but has no memory of where it
#   has been or what corridors it already explored.
#
#   An LSTM fixes this by maintaining a hidden state (h_t, c_t) that acts as
#   a compressed memory of the episode so far.  The cell state c_t can store
#   long-range information (e.g. "I turned left three steps ago") that the
#   hidden state h_t then uses to inform the current decision.
#
#   Self-driving analog: a dashcam that can only see 10 metres ahead.  Without
#   memory you cannot know whether you have already tried a dead-end alley.
#   The LSTM is the driver's short-term memory of the last few seconds of road.
#
# CONCEPT — LSTM vs GRU
#   Both are gated recurrent units.  LSTM uses two vectors (h, c) — a fast
#   working memory (h) and a slow cell tape (c).  GRU merges them into one.
#   LSTM is more expressive; GRU is cheaper.  For maze memory LSTM is standard.
#
# CONCEPT — hidden state lifecycle
#   At the START of every episode: reset h and c to zeros (no prior memory).
#   During the episode:  feed the returned (h, c) back in as input next step.
#   At PPO update time:  re-run the whole episode as one sequence so gradients
#   can flow back through time (BPTT — backpropagation through time).

import torch
import torch.nn as nn
from typing import Optional


class LSTMActorCriticNetwork(nn.Module):
    """Actor-Critic network with an LSTM trunk for partial-observability tasks.

    Architecture
    ------------
    input_projection  : Linear(state_size  → hidden_size) + ReLU
    lstm              : LSTM(hidden_size   → hidden_size, batch_first=True)
    policy_head       : Linear(hidden_size → action_size) → softmax
    value_head        : Linear(hidden_size → 1)

    The sequence dimension fed to the LSTM is always 1 during rollout (one
    step at a time).  During the PPO update the full episode is fed as a
    single sequence so that BPTT gradients traverse the whole trajectory.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size

        # CONCEPT — Input projection
        #   Raw state features live in a very different scale from each other
        #   (normalised x/y are 0-1; wall flags are 0 or 1).  A learned linear
        #   projection + ReLU maps them into a common feature space before the
        #   LSTM sees them, giving the recurrent layer better-conditioned input.
        self.input_projection = nn.Linear(state_size, hidden_size)
        self.relu             = nn.ReLU()

        # CONCEPT — LSTM layer
        #   batch_first=True means input shape is [batch, seq_len, input_size].
        #   At rollout time:   [1, 1, hidden_size]  (batch=1, seq_len=1)
        #   At update time:    [1, T, hidden_size]  (batch=1, seq_len=episode_length)
        #   The LSTM returns (output, (h_n, c_n)).
        #     output: [batch, seq_len, hidden_size] — feature vector per timestep
        #     h_n   : [num_layers, batch, hidden_size] — final hidden state
        #     c_n   : [num_layers, batch, hidden_size] — final cell state
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Policy head (actor) — action distribution over the grid's four moves.
        self.policy_head = nn.Linear(hidden_size, action_size)

        # Value head (critic) — scalar baseline V(s) used to compute advantages.
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state_tensor: torch.Tensor,
        hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one step (or one full-episode sequence) through the network.

        Parameters
        ----------
        state_tensor : shape [batch, state_size]
            Normalised state features.  At rollout time batch=1.
            At update time the caller stacks the full episode along the batch
            dimension and passes it as a sequence — see train_lstm_ppo.py.
        hidden_state : (h_n, c_n) or None
            Previous recurrent state.  Pass None at the start of every episode.

        Returns
        -------
        action_probs   : [batch, action_size]  — probability distribution over actions
        value          : [batch, 1]            — critic estimate V(s)
        new_hidden_state : (h_n, c_n)          — updated recurrent state to carry forward
        """
        # Project raw features into the LSTM's working dimension.
        projected_features = self.relu(self.input_projection(state_tensor))

        # CONCEPT — Sequence dimension for LSTM
        #   nn.LSTM expects [batch, seq_len, features].  At rollout time we pass
        #   one step at a time, so we insert seq_len=1 via unsqueeze(1).
        #   At update time the caller already provides a sequence, so unsqueeze
        #   adds seq_len=1 around each individual state — then we treat the batch
        #   dimension as seq_len by transposing.  The simpler design here is:
        #   always unsqueeze(1) so shape becomes [batch, 1, hidden_size], feed
        #   the whole batch to the LSTM, and squeeze back.  This is correct for
        #   the per-step rollout case.  The full-episode BPTT update passes the
        #   entire episode stacked as [1, T, hidden_size] so the caller must
        #   reshape before calling forward; see ppo_update() in train_lstm_ppo.py.
        lstm_input = projected_features.unsqueeze(1)  # [batch, 1, hidden_size]

        # Resolve initial hidden state — zeros if start of episode.
        resolved_hidden = hidden_state if hidden_state is not None else self.get_initial_hidden_state(
            batch_size=state_tensor.shape[0]
        )

        lstm_output, new_hidden_state = self.lstm(lstm_input, resolved_hidden)
        # lstm_output: [batch, 1, hidden_size] → squeeze seq_len away
        lstm_features = lstm_output.squeeze(1)  # [batch, hidden_size]

        action_probs = torch.softmax(self.policy_head(lstm_features), dim=-1)
        value        = self.value_head(lstm_features)

        return action_probs, value, new_hidden_state

    def get_initial_hidden_state(
        self, batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised (h_0, c_0) for the start of an episode.

        Shape convention matches nn.LSTM: [num_layers, batch, hidden_size].
        num_layers=1 because this network has a single LSTM layer.
        """
        # CONCEPT — Zero initialisation
        #   At episode start the agent has seen nothing.  Zeros mean "no prior
        #   belief" — a neutral slate.  Learned initial states are possible but
        #   add parameters without much benefit for episodic tasks like maze navigation.
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        cell   = torch.zeros(1, batch_size, self.hidden_size)
        return (hidden, cell)
