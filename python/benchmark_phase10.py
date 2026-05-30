# python/benchmark_phase10.py
#
# PURPOSE: Compare all Phase 10 architectures against the Phase 5 MLP PPO baseline
#          on 100 fresh test mazes. Reports success rate, path quality, and inference time.
#
# CORE CONCEPT — What "success" means for each model type
#   DQN / PPO (online RL): trained to reach the goal via trial and error.
#     Success = agent reaches goal within MAX_STEPS.
#   Decision Transformer: predicts one action per step given context.
#     Success = sequence of predicted actions leads to the goal.
#   Diffuser: generates a complete path from start to goal in one shot.
#     Success = generated path contains valid waypoints that end near the goal.
#     Note: diffuser paths may pass through walls (model not trained with hard constraints).
#     We measure both: raw success rate AND collision-free success rate.
#
# CORE CONCEPT — Inference procedure per model type
#
#   MLP PPO (baseline):
#     state = [x/w, y/h, wall_up, wall_down, wall_left, wall_right]
#     action = argmax(policy(state))    — greedy, no sampling at test time
#
#   Transformer PPO:
#     state = GridEncoder.build_grid_tensor(env, x, y)
#     action = argmax(policy(state))    — same procedure, richer state
#
#   Decision Transformer:
#     context = last K (RTG, state, action) triplets
#     action_logits = model(context)   — from state token at current position
#     action = argmax(action_logits[:, -1, :])   — only use the latest prediction
#     RTG decremented by reward received each step
#
#   Diffuser:
#     Generated path = sample_trajectory(denoiser, conditioning, schedule)
#     Map waypoints to actions: for each consecutive pair of waypoints,
#       action = direction from waypoint[t] to waypoint[t+1] (UP/DOWN/LEFT/RIGHT)
#     Execute action sequence in the environment, count collisions
#
# METRICS:
#   success_rate:      fraction of mazes where agent reached goal
#   mean_path_length:  steps taken on successful episodes
#   inference_time_ms: wall-clock time per step (policy) or per path (diffuser)
#   collision_rate:    for Diffuser only — fraction of steps hitting walls

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import pathplanning
from networks.actor_critic         import ActorCriticNetwork
from networks.transformer_policy   import TransformerActorCritic
from networks.decision_transformer import DecisionTransformer, CONTEXT_LENGTH, STATE_DIM
from networks.trajectory_diffuser  import (TrajectoryDenoiser, build_diffusion_schedule,
                                            MAX_PATH_LENGTH, DIFFUSION_TIMESTEPS)
from networks.grid_encoder         import GridEncoder
from train_decision_transformer    import extract_state_vector
from train_diffuser                import sample_trajectory

# ---- Configuration ----------------------------------------------------------

GRID_HEIGHT      = 41
GRID_WIDTH       = 41
OBSTACLE_DENSITY = 0.25
MAX_STEPS        = GRID_HEIGHT * GRID_WIDTH * 4
NUM_TEST_MAZES   = 100
TEST_SEED_OFFSET = 5000     # separate from training seeds (0-1999) and Phase 9 (1000-1099)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_mlp_ppo_baseline() -> ActorCriticNetwork:
    """
    Implement:
    1. network = ActorCriticNetwork(state_size=6, action_size=4, hidden_size=128)
    2. weights_path = os.path.join(DATA_DIR, 'ppo_model.pt')
       If file exists: network.load_state_dict(torch.load(weights_path))
       Else: print warning "MLP PPO weights not found — using random initialisation"
    3. network.eval()
    4. return network
    """
    network      = ActorCriticNetwork(state_size=6, action_size=4, hidden_size=128)
    weights_path = os.path.join(DATA_DIR, 'ppo_model.pt')
    if os.path.exists(weights_path):
        network.load_state_dict(torch.load(weights_path, weights_only=True))
    else:
        print("MLP PPO weights not found — using random initialisation")
    network.eval()
    return network


def load_transformer_policy() -> TransformerActorCritic:
    """
    Implement:
    1. network = TransformerActorCritic(GRID_HEIGHT, GRID_WIDTH)
    2. Load from python/data/transformer_policy.pt if it exists
    3. network.eval()
    4. return network
    """
    network      = TransformerActorCritic(GRID_HEIGHT, GRID_WIDTH)
    weights_path = os.path.join(DATA_DIR, 'transformer_policy.pt')
    if os.path.exists(weights_path):
        network.load_state_dict(torch.load(weights_path, weights_only=True))
    else:
        print("Transformer policy weights not found — using random initialisation")
    network.eval()
    return network


def load_decision_transformer() -> DecisionTransformer:
    """
    Implement:
    1. model = DecisionTransformer()
    2. Load from python/data/decision_transformer.pt if it exists
    3. model.eval()
    4. return model
    """
    model        = DecisionTransformer()
    weights_path = os.path.join(DATA_DIR, 'decision_transformer.pt')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    else:
        print("Decision Transformer weights not found — using random initialisation")
    model.eval()
    return model


def load_diffuser() -> tuple:
    """
    Returns (denoiser, schedule) tuple.
    Implement:
    1. saved = torch.load(os.path.join(DATA_DIR, 'trajectory_diffuser.pt'))
    2. denoiser = TrajectoryDenoiser()
       denoiser.load_state_dict(saved['model_state'])
       denoiser.eval()
    3. schedule = build_diffusion_schedule()  — rebuild from constants (matches training)
    4. return (denoiser, schedule)
    """
    saved    = torch.load(os.path.join(DATA_DIR, 'trajectory_diffuser.pt'), weights_only=False)
    denoiser = TrajectoryDenoiser()
    denoiser.load_state_dict(saved['model_state'])
    denoiser.eval()
    schedule = build_diffusion_schedule()
    return (denoiser, schedule)


def evaluate_step_policy(
    model:         torch.nn.Module,
    environment:   pathplanning.GridEnvironment,
    use_grid_state: bool = False
) -> tuple:
    """
    CONCEPT — Greedy rollout for step-based policies (PPO variants):
    At each step, ask the policy for the best action. Execute it. Repeat.
    At test time we use argmax (greedy) not sampling — deterministic and comparable.

    Implement:
    1. state = env.reset()  — [x, y]
    2. For step_count in range(MAX_STEPS):
       a. Build state tensor:
            if use_grid_state: tensor = GridEncoder.build_grid_tensor(env, x, y, ...)
            else: tensor = torch.FloatTensor([x/W, y/H, ...los...]).unsqueeze(0)
       b. with torch.no_grad(): action_probs, _ = model(tensor)
       c. action = int(torch.argmax(action_probs, dim=-1))
       d. result = env.step(action)
       e. If result[3] > 0.5: return (True, step_count + 1, time_ms)
    3. return (False, MAX_STEPS, time_ms)
    """
    state           = environment.reset()
    current_x, current_y = int(state[0]), int(state[1])
    start_time      = time.perf_counter()
    total_inference = 0.0

    for step_count in range(MAX_STEPS):
        if use_grid_state:
            state_tensor = GridEncoder.build_grid_tensor(
                environment, current_x, current_y, GRID_HEIGHT, GRID_WIDTH)
        else:
            los          = environment.getLineOfSight(current_x, current_y)
            state_tensor = torch.FloatTensor([
                current_x / GRID_WIDTH, current_y / GRID_HEIGHT,
                los[0], los[1], los[2], los[3]
            ]).unsqueeze(0)

        infer_start = time.perf_counter()
        with torch.no_grad():
            action_probs, _ = model(state_tensor)
        total_inference += (time.perf_counter() - infer_start) * 1000.0

        action = int(torch.argmax(action_probs, dim=-1))
        result = environment.step(action)
        current_x, current_y = int(result[0]), int(result[1])

        if result[3] > 0.5:
            avg_inference_ms = total_inference / (step_count + 1)
            return (True, step_count + 1, avg_inference_ms)

    avg_inference_ms = total_inference / MAX_STEPS
    return (False, MAX_STEPS, avg_inference_ms)


def evaluate_decision_transformer(
    model:        DecisionTransformer,
    environment:  pathplanning.GridEnvironment
) -> tuple:
    """
    CONCEPT — Autoregressive action generation with RTG conditioning:
    Maintain a rolling context window of the last K (RTG, state, action) triplets.
    At each step, feed the context to the model, get action logit for the current position,
    execute the action, update RTG and context.

    Key detail: the context window is FIXED SIZE (K=20). For the first K steps,
    pad the beginning with zeros. After K steps, shift the window forward.

    Implement:
    1. desired_rtg = 80.0 / 100.0  (normalised — asking for near-optimal performance)
    2. Initialise empty context buffers (K, 1), (K, 8), (K,), (K,) with zeros
    3. current_rtg = desired_rtg
    4. For step_count in range(MAX_STEPS):
       a. state_vector = extract_state_vector(env, x, y)
       b. Shift context window: append current (rtg, state, action, timestep)
          and drop the oldest. For first steps: fill from the end backward.
       c. model_input = build tensors from context window
       d. with torch.no_grad(): logits = model(rtg_seq, state_seq, action_seq, timestep_seq)
       e. action = int(torch.argmax(logits[0, -1, :]))  — last position in window
       f. result = env.step(action)
       g. current_rtg -= result[2] / 100.0  (update RTG by actual reward, normalised)
       h. If done: return (True, step_count + 1, time_ms)
    5. return (False, MAX_STEPS, time_ms)
    """
    state             = environment.reset()
    current_x, current_y = int(state[0]), int(state[1])
    current_rtg       = 80.0 / 100.0
    start_time        = time.perf_counter()

    rtg_context    = torch.zeros(CONTEXT_LENGTH, 1)
    state_context  = torch.zeros(CONTEXT_LENGTH, STATE_DIM)
    action_context = torch.zeros(CONTEXT_LENGTH, dtype=torch.long)
    time_context   = torch.zeros(CONTEXT_LENGTH, dtype=torch.long)

    for step_count in range(MAX_STEPS):
        state_vector = extract_state_vector(environment, current_x, current_y)

        rtg_context    = torch.roll(rtg_context,    -1, dims=0)
        state_context  = torch.roll(state_context,  -1, dims=0)
        action_context = torch.roll(action_context, -1, dims=0)
        time_context   = torch.roll(time_context,   -1, dims=0)

        rtg_context[-1]    = current_rtg
        state_context[-1]  = torch.FloatTensor(state_vector)
        action_context[-1] = 0
        time_context[-1]   = step_count

        with torch.no_grad():
            logits = model(
                rtg_context.unsqueeze(0),
                state_context.unsqueeze(0),
                action_context.unsqueeze(0),
                time_context.unsqueeze(0)
            )
        action = int(torch.argmax(logits[0, -1, :]))

        action_context[-1] = action

        result             = environment.step(action)
        current_x, current_y = int(result[0]), int(result[1])
        reward             = result[2]
        done               = result[3] > 0.5

        current_rtg -= reward / 100.0

        if done:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return (True, step_count + 1, elapsed_ms)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return (False, MAX_STEPS, elapsed_ms)


def evaluate_diffuser(
    denoiser:     TrajectoryDenoiser,
    schedule:     dict,
    environment:  pathplanning.GridEnvironment
) -> tuple:
    """
    CONCEPT — Executing a generated trajectory:
    The diffuser generates a (MAX_PATH_LENGTH, 2) sequence of normalised waypoints.
    We convert consecutive waypoint pairs to grid actions (UP/DOWN/LEFT/RIGHT).
    If the generated waypoint is not a valid adjacent cell, we use the nearest valid action.
    This is "open-loop" execution — the agent follows the plan without re-planning.

    Implement:
    1. goal = env.getGoal()  — [gx, gy]
       conditioning = torch.tensor([[1/GRID_WIDTH, 1/GRID_HEIGHT,
                                     goal[0]/GRID_WIDTH, goal[1]/GRID_HEIGHT]])
    2. start_time = time.perf_counter()
       generated_waypoints = sample_trajectory(denoiser, conditioning, schedule)
       inference_time_ms = (time.perf_counter() - start_time) * 1000

    3. Execute each waypoint as a step:
       current_x, current_y = 1, 1  (start position)
       For each waypoint (wx_norm, wy_norm) in generated_waypoints:
           wx = round(wx_norm * GRID_WIDTH)
           wy = round(wy_norm * GRID_HEIGHT)
           dx = wx - current_x
           dy = wy - current_y
           action = nearest_action(dx, dy)  — UP/DOWN/LEFT/RIGHT based on sign of dx, dy
           result = env.step(action)
           if done: return (True, step_count+1, inference_time_ms, collisions)
           if env.isObstacle(new_x, new_y): collisions += 1

    4. return (False, MAX_PATH_LENGTH, inference_time_ms, collisions)
    """
    goal         = environment.getGoal()
    conditioning = torch.tensor([[
        1.0 / GRID_WIDTH,  1.0 / GRID_HEIGHT,
        goal[0] / GRID_WIDTH, goal[1] / GRID_HEIGHT,
    ]])

    start_time          = time.perf_counter()
    generated_waypoints = sample_trajectory(denoiser, conditioning, schedule)
    inference_time_ms   = (time.perf_counter() - start_time) * 1000.0

    environment.reset()
    current_x, current_y = 1, 1
    collisions            = 0

    for step_count, (wx_norm, wy_norm) in enumerate(generated_waypoints):
        wx = int(round(float(wx_norm) * GRID_WIDTH))
        wy = int(round(float(wy_norm) * GRID_HEIGHT))

        dx = wx - current_x
        dy = wy - current_y

        if abs(dx) >= abs(dy):
            action = 3 if dx >= 0 else 2
        else:
            action = 1 if dy >= 0 else 0

        intended_x = current_x + (1 if action == 3 else -1 if action == 2 else 0)
        intended_y = current_y + (1 if action == 1 else -1 if action == 0 else 0)
        if environment.isObstacle(intended_x, intended_y):
            collisions += 1

        result              = environment.step(action)
        current_x, current_y = int(result[0]), int(result[1])
        done                = result[3] > 0.5

        if done:
            return (True, step_count + 1, inference_time_ms, collisions)

    return (False, MAX_PATH_LENGTH, inference_time_ms, collisions)


def print_benchmark_table(results: dict) -> None:
    """
    CONCEPT — Interpreting Phase 10 results:
    Each model type has a different success/failure mode:
      - MLP PPO: baseline — succeeds on easy mazes, struggles on hard
      - Transformer PPO: should outperform MLP on complex topologies
      - Decision Transformer: relies on in-context pattern matching — may fail on
        mazes very different from training distribution
      - Diffuser: high collision rate expected — generated paths ignore grid topology.
        Its value is path diversity and the paradigm demonstration, not raw success rate.

    Implement:
    1. For each model name and metric set in results:
       Print: name, success_rate (%), mean_path_length, inference_time_ms
    2. Print separator line
    3. Print "Diffuser collision rate: X%" separately (unique metric)
    4. Print note: "DT and Diffuser degrade gracefully — architecture insights matter
       more than benchmark numbers at this stage."
    """
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Success %':>10} {'Mean Steps':>12} {'Infer ms':>10}")
    print("-" * 70)

    for name, metrics in results.items():
        if name == 'Diffuser':
            continue
        successes    = metrics['successes']
        path_lengths = [pl for ok, pl in zip(successes, metrics['path_lengths']) if ok]
        success_rate = 100.0 * sum(successes) / len(successes) if successes else 0.0
        mean_steps   = np.mean(path_lengths) if path_lengths else float('nan')
        mean_time    = np.mean(metrics['times']) if metrics['times'] else float('nan')
        print(f"{name:<25} {success_rate:>9.1f}% {mean_steps:>12.1f} {mean_time:>10.3f}")

    if 'Diffuser' in results:
        d            = results['Diffuser']
        success_rate = 100.0 * sum(d['successes']) / len(d['successes']) if d['successes'] else 0.0
        collision_rate = 100.0 * sum(d['collisions']) / max(sum(d['path_lengths']), 1)
        mean_time    = np.mean(d['times']) if d['times'] else float('nan')
        print(f"{'Diffuser':<25} {success_rate:>9.1f}% {'(open-loop)':>12} {mean_time:>10.1f}")
        print(f"  Diffuser collision rate: {collision_rate:.1f}%")

    print("=" * 70)
    print("Note: DT and Diffuser degrade gracefully — architecture insights matter")
    print("more than benchmark numbers at this stage.")


def main() -> None:
    """
    Implement:
    1. Load all four models (use try/except — some may not be trained yet):
           mlp_ppo = load_mlp_ppo_baseline()
           transformer_ppo = load_transformer_policy()
           decision_transformer = load_decision_transformer()
           denoiser, schedule = load_diffuser()

    2. results = {}
    3. For each test maze (seed = TEST_SEED_OFFSET + maze_index):
       Create GridEnvironment(..., seed=seed)
       Evaluate each model, collect (success, path_length, time_ms)
       Append to results lists

    4. Print progress every 10 mazes.
    5. print_benchmark_table(results)
    """
    mlp_ppo             = load_mlp_ppo_baseline()
    transformer_ppo     = load_transformer_policy()
    decision_transformer = load_decision_transformer()

    try:
        denoiser, schedule = load_diffuser()
        diffuser_available = True
    except Exception as exc:
        print(f"Diffuser not available: {exc}")
        diffuser_available = False

    results = {
        'MLP PPO':              {'successes': [], 'path_lengths': [], 'times': []},
        'Transformer PPO':      {'successes': [], 'path_lengths': [], 'times': []},
        'Decision Transformer': {'successes': [], 'path_lengths': [], 'times': []},
        'Diffuser':             {'successes': [], 'path_lengths': [], 'times': [], 'collisions': []},
    }

    for maze_index in range(NUM_TEST_MAZES):
        seed = TEST_SEED_OFFSET + maze_index

        env = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1,
                                           GRID_WIDTH - 2, GRID_HEIGHT - 2,
                                           OBSTACLE_DENSITY, seed)
        success, steps, time_ms = evaluate_step_policy(mlp_ppo, env, use_grid_state=False)
        results['MLP PPO']['successes'].append(success)
        results['MLP PPO']['path_lengths'].append(steps)
        results['MLP PPO']['times'].append(time_ms)

        env = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1,
                                           GRID_WIDTH - 2, GRID_HEIGHT - 2,
                                           OBSTACLE_DENSITY, seed)
        success, steps, time_ms = evaluate_step_policy(transformer_ppo, env, use_grid_state=True)
        results['Transformer PPO']['successes'].append(success)
        results['Transformer PPO']['path_lengths'].append(steps)
        results['Transformer PPO']['times'].append(time_ms)

        env = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1,
                                           GRID_WIDTH - 2, GRID_HEIGHT - 2,
                                           OBSTACLE_DENSITY, seed)
        success, steps, time_ms = evaluate_decision_transformer(decision_transformer, env)
        results['Decision Transformer']['successes'].append(success)
        results['Decision Transformer']['path_lengths'].append(steps)
        results['Decision Transformer']['times'].append(time_ms)

        if diffuser_available:
            env = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1,
                                               GRID_WIDTH - 2, GRID_HEIGHT - 2,
                                               OBSTACLE_DENSITY, seed)
            success, steps, time_ms, colls = evaluate_diffuser(denoiser, schedule, env)
            results['Diffuser']['successes'].append(success)
            results['Diffuser']['path_lengths'].append(steps)
            results['Diffuser']['times'].append(time_ms)
            results['Diffuser']['collisions'].append(colls)

        if (maze_index + 1) % 10 == 0:
            print(f"Progress: {maze_index + 1}/{NUM_TEST_MAZES} mazes")

    if not diffuser_available:
        del results['Diffuser']

    print_benchmark_table(results)


if __name__ == '__main__':
    main()
