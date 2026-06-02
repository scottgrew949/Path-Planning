# python/train_alphazero.py
# AlphaZero-style training loop: MCTS guided by a neural network, trained
# on the visit distributions and outcomes MCTS itself produces.
#
# CONCEPT — The AlphaZero feedback loop
#   Classical MCTS uses random rollouts to estimate node value — expensive and
#   noisy for large state spaces.  AlphaZero replaces rollouts with a neural
#   network that provides two signals at every node:
#     1. Prior probability P(s,a): steers tree expansion toward promising actions.
#     2. Value estimate V(s): backs up immediately without simulating to a terminal.
#   The network is then *trained on the very distributions MCTS produces*, which
#   are more accurate than the raw network output because MCTS combines many
#   simulated paths.  This creates a self-improving loop:
#     better network → better MCTS → better training targets → better network.
#
# CONCEPT — Why single-agent AlphaZero still works
#   Original AlphaZero flips the value sign on every backup because two
#   opponents alternate turns — what is good for one is bad for the other.
#   In a single-agent grid problem there is no opponent, so value sign never
#   flips during backup.  The rest of the algorithm is identical.
#
# CONCEPT — Temperature in action selection
#   Early in an episode we want exploration: sample from the MCTS visit
#   distribution (temperature = 1, full stochasticity).
#   Later we want exploitation: take the most-visited action (temperature → 0,
#   i.e. greedy / argmax).  The crossover point TEMPERATURE_THRESHOLD is a
#   hyperparameter tuned by feel for the grid size.
#
# CONCEPT — Why we track position manually during MCTS simulations
#   The pybind11 env does not expose clone().  Calling env.step() during MCTS
#   would permanently mutate the real environment state.  Instead, simulations
#   use a lightweight position-only state (x, y) and advance it with
#   _simulate_action(), which checks walls via env.getLineOfSight() and treats
#   out-of-bounds or obstacle moves as no-ops (position unchanged).
#   The network value estimate replaces rollouts entirely — we never need to
#   actually reach a terminal node during simulation.
#
# Self-driving analog:
#   MCTS simulations = mental lookahead the planner runs before committing to a
#   steering action.  The network priors focus the search on lanes that look
#   safe; the value head says "this intersection geometry is 0.8 / 1.0 toward
#   reaching the destination."

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
import torch
import torch.nn as nn
import pathplanning
from networks.alphazero_network import AlphaZeroNetwork

# ---- Hyperparameters ---------------------------------------------------------

GRID_WIDTH           = 41
GRID_HEIGHT          = 41
STATE_SIZE           = 6        # [x/w, y/h, wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE          = 4        # 0=UP 1=DOWN 2=LEFT 3=RIGHT
HIDDEN_SIZE          = 128

NUM_EPISODES         = 500
NUM_SIMULATIONS      = 50       # MCTS simulations per move
EXPLORATION_CONSTANT = 1.414    # C in UCB: controls exploration vs exploitation
LEARNING_RATE        = 0.001
BATCH_SIZE           = 32
REPLAY_BUFFER_CAP    = 10_000   # max stored training examples

MAX_STEPS_PER_EPISODE = GRID_WIDTH * GRID_HEIGHT * 4

# Step threshold after which we switch from stochastic to greedy action selection.
TEMPERATURE_THRESHOLD = MAX_STEPS_PER_EPISODE // 4

# Action deltas: UP, DOWN, LEFT, RIGHT
ACTION_DELTAS = {
    0: ( 0, -1),   # UP    — decreasing y
    1: ( 0,  1),   # DOWN  — increasing y
    2: (-1,  0),   # LEFT  — decreasing x
    3: ( 1,  0),   # RIGHT — increasing x
}

# ---- State helper ------------------------------------------------------------

def state_to_tensor(state: list, env) -> torch.Tensor:
    # CONCEPT — Why we normalise position but not wall flags
    #   Position is continuous in [0, width/height]; normalising to [0,1]
    #   prevents large magnitude differences from dominating the first linear
    #   layer.  Wall flags are already binary {0,1} — no normalisation needed.
    normalised = [
        state[0] / GRID_WIDTH,
        state[1] / GRID_HEIGHT,
    ] + list(env.getLineOfSight(state[0], state[1]))
    return torch.FloatTensor(normalised).unsqueeze(0)


# ---- Simulated step (position-only, no env mutation) -------------------------

def _simulate_action(position: tuple, action: int, env) -> tuple:
    """Advance position by one action without mutating env state.

    Returns the new (x, y) — same as input if the move hits a wall or
    boundary, matching the real env's no-op behaviour for blocked moves.

    CONCEPT — Wall encoding from getLineOfSight
      getLineOfSight(x, y) returns [wall_up, wall_down, wall_left, wall_right]
      where 1.0 means the direction IS blocked.  We check the wall in the
      direction of the intended move before committing to it, so we never
      incorrectly enter an obstacle cell during simulation.
    """
    current_x, current_y = position
    delta_x, delta_y     = ACTION_DELTAS[action]
    candidate_x          = current_x + delta_x
    candidate_y          = current_y + delta_y

    # Boundary check — env gives no inBounds() so check manually.
    if candidate_x < 0 or candidate_x >= GRID_WIDTH:
        return position
    if candidate_y < 0 or candidate_y >= GRID_HEIGHT:
        return position

    # Wall check via line-of-sight: index matches ACTION_DELTAS order.
    walls = env.getLineOfSight(current_x, current_y)
    if walls[action] > 0.5:
        return position

    return (candidate_x, candidate_y)


# ---- MCTSNode ----------------------------------------------------------------

class MCTSNode:
    """One node in the MCTS search tree, corresponding to a single (x, y) state.

    CONCEPT — Why we store prior_probability per node rather than per edge
      Classic MCTS stores statistics on edges (parent → child action).
      AlphaZero DeepMind stores P(s,a) on the child node itself — the node
      represents the state reached by taking that action from its parent.
      Both are equivalent; storing on the child simplifies UCB calculation
      since all required data is local to the child node.
    """

    def __init__(
        self,
        state: tuple,
        parent: "MCTSNode | None" = None,
        action_taken: int | None = None,
        prior_probability: float = 0.0,
    ):
        self.state             = state            # (x, y)
        self.parent            = parent
        self.action_taken      = action_taken     # action that led here from parent
        self.prior_probability = prior_probability

        self.children:    dict[int, "MCTSNode"] = {}
        self.visit_count: int   = 0
        self.value_sum:   float = 0.0
        self.is_expanded: bool  = False

    def value(self) -> float:
        """Mean value over all backups through this node (0 if unvisited)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, exploration_constant: float) -> float:
        """Upper Confidence Bound score used for child selection.

        CONCEPT — UCB formula in AlphaZero
          UCB(s, a) = Q(s,a) + C * P(s,a) * sqrt(N(parent)) / (1 + N(child))
          Q = mean backed-up value      — exploitation term
          P = prior from the network    — focuses search on network-preferred actions
          sqrt(N_parent)/(1+N_child)    — exploration bonus, decays as child is visited

          Without the prior P this degenerates to classical UCT.
          The network prior is the key AlphaZero contribution: it biases the
          exploration bonus toward actions the network already thinks are good,
          so the tree is not wasted on obviously bad branches early in training.
        """
        if self.parent is None:
            return 0.0
        exploration_bonus = (
            exploration_constant
            * self.prior_probability
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return self.value() + exploration_bonus

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        """Return the child node with the highest UCB score."""
        return max(
            self.children.values(),
            key=lambda child_node: child_node.ucb_score(exploration_constant),
        )

    def expand(self, action_probs: list[float]) -> None:
        """Create one child node per action, using network priors as prior_probability.

        CONCEPT — Expanding all children at once vs one at a time
          Classical UCT expands one new child per simulation (random unvisited).
          AlphaZero expands all children simultaneously because it has prior
          probabilities from the network for every action.  Expanding all at
          once ensures that even low-prior actions remain reachable — the UCB
          formula will still visit them eventually if promising.
        """
        for action_index in range(ACTION_SIZE):
            if action_index not in self.children:
                self.children[action_index] = MCTSNode(
                    state             = self.state,   # position updated lazily in search
                    parent            = self,
                    action_taken      = action_index,
                    prior_probability = float(action_probs[action_index]),
                )
        self.is_expanded = True


# ---- AlphaZeroMCTS -----------------------------------------------------------

class AlphaZeroMCTS:
    """MCTS search loop driven by an AlphaZeroNetwork.

    CONCEPT — Why MCTS improves on raw network output
      The network gives a single-shot policy estimate P(s,a) for the current
      state.  MCTS runs NUM_SIMULATIONS forward simulations from that state,
      each time selecting actions via UCB, expanding leaves with network priors,
      and backing up the network's value estimate.  The resulting visit count
      distribution is much more accurate than the raw prior because it averages
      across many simulated trajectories.  This improved estimate then becomes
      the training target — the network is asked to match visit counts, not just
      its own initial guess.
    """

    def __init__(
        self,
        network:              AlphaZeroNetwork,
        num_simulations:      int   = 50,
        exploration_constant: float = 1.414,
        grid_width:           int   = 41,
        grid_height:          int   = 41,
        goal_x:               int   = 38,
        goal_y:               int   = 40,
    ):
        self.network              = network
        self.num_simulations      = num_simulations
        self.exploration_constant = exploration_constant
        self.grid_width           = grid_width
        self.grid_height          = grid_height
        self.goal_position        = (goal_x, goal_y)

    def _is_terminal(self, position: tuple) -> bool:
        return position == self.goal_position

    def _get_network_output(self, position: tuple, env) -> tuple[list[float], float]:
        """Run the network for a given position and return (policy_probs, value)."""
        state_list   = [position[0], position[1]]
        state_tensor = state_to_tensor(state_list, env)
        with torch.no_grad():
            policy_tensor, value_tensor = self.network(state_tensor)
        policy_probs = policy_tensor.squeeze(0).tolist()
        value        = value_tensor.item()
        return policy_probs, value

    def search(self, root_state: list, env) -> list[float]:
        """Run MCTS from root_state and return a policy vector.

        The policy vector is the normalised visit count distribution:
          pi(a | s) = N(s, a) / sum_a N(s, a)
        This is more accurate than the raw network prior and becomes the
        training target for the policy head.

        Returns a list of length ACTION_SIZE with probabilities summing to 1.
        """
        root_position   = (int(root_state[0]), int(root_state[1]))
        root_node       = MCTSNode(state=root_position)

        # Expand the root immediately so best_child() has children to choose from.
        root_policy_probs, _ = self._get_network_output(root_position, env)
        root_node.expand(root_policy_probs)
        root_node.visit_count = 1   # count the root visit

        for _ in range(self.num_simulations):
            node     = root_node
            position = root_position

            # ---- Selection ---------------------------------------------------
            # Descend the tree following best_child until we reach a leaf.
            # A leaf is either unexpanded or a terminal state.
            while node.is_expanded and not self._is_terminal(position):
                node          = node.best_child(self.exploration_constant)
                position      = _simulate_action(position, node.action_taken, env)
                node.state    = position   # update position lazily after the move

            # ---- Expansion + Evaluation --------------------------------------
            if self._is_terminal(position):
                # Terminal node: value is a guaranteed win (+1).
                simulation_value = 1.0
            else:
                # Non-terminal leaf: ask the network for prior + value estimate.
                leaf_policy_probs, simulation_value = self._get_network_output(
                    position, env
                )
                if not node.is_expanded:
                    node.expand(leaf_policy_probs)

            # ---- Backup ------------------------------------------------------
            # Propagate simulation_value up to the root.
            # Single-agent: no sign flip (no opponent perspective to alternate).
            backup_node = node
            while backup_node is not None:
                backup_node.visit_count += 1
                backup_node.value_sum   += simulation_value
                backup_node              = backup_node.parent

        # ---- Policy from visit counts ----------------------------------------
        # CONCEPT — Why visit counts not Q-values as training target
        #   Q-values are noisy averages; visit counts reflect how many times
        #   MCTS decided each action was worth exploring.  A high-prior action
        #   that also has high backed-up value will accumulate visits quickly —
        #   the count distribution is a smoothed, robust policy estimate.
        visit_counts = [
            root_node.children[action_index].visit_count
            if action_index in root_node.children else 0
            for action_index in range(ACTION_SIZE)
        ]
        total_visits = sum(visit_counts) or 1   # avoid division by zero
        normalised_policy = [count / total_visits for count in visit_counts]
        return normalised_policy


# ---- Self-play episode -------------------------------------------------------

def self_play_episode(
    network:    AlphaZeroNetwork,
    mcts:       AlphaZeroMCTS,
    env,
    max_steps:  int,
    goal_x:     int,
    goal_y:     int,
) -> tuple[list, bool]:
    """Run one episode of self-play and collect training examples.

    Returns:
        training_examples: list of (state_tensor, policy_target, value_target)
            value_target is filled at the end once the outcome is known.
        goal_reached: bool

    CONCEPT — Deferred value targets
      During the episode we don't know the outcome yet, so we store
      (state_tensor, policy_target) pairs and fill in the value_target
      retroactively at the end.  +1 if the goal was reached, -1 if the
      episode timed out.  This matches how AlphaZero labels game outcomes:
      winner gets +1, loser -1, applied consistently to every position in
      the game history.

    CONCEPT — Temperature schedule
      Steps < TEMPERATURE_THRESHOLD: sample from MCTS distribution (explore).
      Steps >= TEMPERATURE_THRESHOLD: argmax of visit distribution (exploit).
      Early exploration prevents the agent from prematurely committing to
      suboptimal routes; late exploitation improves value target quality.
    """
    state        = env.reset()
    goal_reached = False

    # Buffer for (state_tensor, policy_target) — value filled after episode.
    partial_examples: list[tuple[torch.Tensor, list[float]]] = []

    for step_number in range(max_steps):
        state_tensor = state_to_tensor(state, env)
        mcts_policy  = mcts.search(state, env)

        partial_examples.append((state_tensor, mcts_policy))

        # Action selection: stochastic early, greedy later.
        if step_number < TEMPERATURE_THRESHOLD:
            action = random.choices(range(ACTION_SIZE), weights=mcts_policy)[0]
        else:
            action = int(max(range(ACTION_SIZE), key=lambda action_idx: mcts_policy[action_idx]))

        result     = env.step(action)
        state      = [int(result[0]), int(result[1])]
        done       = result[3] > 0.5

        if done:
            # Check if terminal because goal reached or because step limit in env.
            goal_reached = (state[0] == goal_x and state[1] == goal_y)
            break

    # Assign outcome to every collected example.
    episode_outcome = 1.0 if goal_reached else -1.0
    training_examples = [
        (state_tensor, policy_target, episode_outcome)
        for state_tensor, policy_target in partial_examples
    ]
    return training_examples, goal_reached


# ---- Training ----------------------------------------------------------------

def train() -> None:
    """AlphaZero self-play training loop.

    CONCEPT — Why we mix self-play with gradient updates
      AlphaZero alternates self-play (data generation) with network updates.
      We use a simplified interleaved version: after every episode, if the
      replay buffer has enough examples, we sample a mini-batch and update.
      This avoids a hard boundary between generation and training phases,
      which works well for small grids where episodes are short.

    CONCEPT — Replay buffer in AlphaZero
      Original AlphaZero stored the most recent ~500k positions.  We cap at
      REPLAY_BUFFER_CAP.  Older examples are less relevant because the policy
      that generated them has changed, but retaining some history stabilises
      training by decorrelating mini-batches (same motivation as DQN).
    """
    print("\n" + "=" * 60)
    print("AlphaZero — MCTS + Neural Policy/Value")
    print("=" * 60)
    print("Plain MCTS (menu option 1) guides tree search with random rollouts.")
    print("AlphaZero replaces both the rollout AND the expansion heuristic")
    print("with a single neural network:")
    print("  Policy head: prior probability over actions — biases UCB exploration")
    print("               toward moves the network already believes are good.")
    print("  Value head:  immediate state value — no rollout to terminal needed.")
    print("")
    print("Self-play: the agent plays against itself, collecting (state, MCTS policy,")
    print("outcome) tuples. The network trains on these — it learns to predict the")
    print("MCTS visit distribution AND the episode outcome from any position.")
    print("")
    print("Self-driving analog: the policy head is route intuition (which direction")
    print("looks promising), the value head is hazard assessment (how safe is")
    print("this position). MCTS refines both with lookahead before committing.")
    print("")
    print("What to watch: goal_reached rate climbing as the network improves.")
    print("Early episodes: MCTS explores near-randomly. Late: it follows learned priors.")
    print("")
    print("  Episode      = one self-play game (start → goal or timeout)")
    print("  Goal reached = did the agent reach the goal this episode?")
    print("  Examples     = total (state, policy, value) training tuples collected")
    print("")

    goal_x = 38
    goal_y = 40

    env     = pathplanning.GridEnvironment(
        GRID_WIDTH, GRID_HEIGHT, 0, 0, goal_x, goal_y, 0.3
    )
    network   = AlphaZeroNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    mcts      = AlphaZeroMCTS(
        network              = network,
        num_simulations      = NUM_SIMULATIONS,
        exploration_constant = EXPLORATION_CONSTANT,
        grid_width           = GRID_WIDTH,
        grid_height          = GRID_HEIGHT,
        goal_x               = goal_x,
        goal_y               = goal_y,
    )
    optimizer     = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    replay_buffer: list[tuple[torch.Tensor, list[float], float]] = []

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn  = nn.MSELoss()

    for episode_number in range(1, NUM_EPISODES + 1):

        training_examples, goal_reached = self_play_episode(
            network, mcts, env, MAX_STEPS_PER_EPISODE, goal_x, goal_y
        )

        # CONCEPT — Buffer management: drop oldest examples when at capacity.
        replay_buffer.extend(training_examples)
        if len(replay_buffer) > REPLAY_BUFFER_CAP:
            excess = len(replay_buffer) - REPLAY_BUFFER_CAP
            del replay_buffer[:excess]

        if len(replay_buffer) < BATCH_SIZE:
            if episode_number % 10 == 0:
                print(
                    f"Episode {episode_number:4d} | "
                    f"Goal: {'YES' if goal_reached else 'no ':3s} | "
                    f"Buffer: {len(replay_buffer):5d} (filling...)"
                )
            continue

        # ---- Sample mini-batch -----------------------------------------------
        batch = random.sample(replay_buffer, BATCH_SIZE)

        batch_states        = torch.cat([example[0] for example in batch])
        batch_policy_targets = torch.FloatTensor([example[1] for example in batch])
        batch_value_targets  = torch.FloatTensor(
            [[example[2]] for example in batch]
        )

        # ---- Forward pass ----------------------------------------------------
        network_policies, network_values = network(batch_states)

        # CONCEPT — Policy loss: cross-entropy between network output and MCTS visit distribution
        #   MCTS visit distribution is a soft label (probabilities), not a hard class.
        #   Cross-entropy with soft targets = -sum(target * log(prediction)).
        #   torch.nn.CrossEntropyLoss expects class indices (hard), so we
        #   compute the soft cross-entropy manually.
        log_policy_probs = torch.log(network_policies + 1e-8)
        policy_loss = -(batch_policy_targets * log_policy_probs).sum(dim=1).mean()

        # Value loss: MSE between predicted tanh value and episode outcome {+1, -1}.
        value_loss = value_loss_fn(network_values, batch_value_targets)

        total_loss = policy_loss + value_loss

        # ---- Gradient update -------------------------------------------------
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode_number % 10 == 0:
            print(
                f"Episode {episode_number:4d} | "
                f"Goal: {'YES' if goal_reached else 'no ':3s} | "
                f"Examples: {len(replay_buffer):5d} | "
                f"PolicyLoss: {policy_loss.item():.4f} | "
                f"ValueLoss: {value_loss.item():.4f}"
            )


if __name__ == "__main__":
    train()
