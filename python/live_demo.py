# python/live_demo.py
# Live simulation: agent navigates a maze with fog-of-war and patrolling obstacles.
# Uses D* Lite for replanning. Fully interactive.
#
# Self-driving car analog:
#   Agent     = vehicle with limited sensor range
#   Fog       = LiDAR range limit
#   Obstacles = slow vehicles blocking the corridor — agent must wait or replan
#   D* Lite   = incremental replanner
#
# Run: source venv/bin/activate && python python/live_demo.py
# Controls: left-click = toggle wall | space = pause | R = reset | slider = speed

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import pathplanning

# --- Display ------------------------------------------------------------------
FREE      = 0
WALL      = 1
DYNAMIC   = 2
PATH      = 3
AGENT     = 4
GOAL_CELL = 5
UNKNOWN   = 6

COLORMAP = mcolors.ListedColormap([
    "#e8e8e8",  # free
    "#1a1a2e",  # wall
    "#ff6600",  # dynamic obstacle
    "#4fc3f7",  # planned path
    "#00e676",  # agent
    "#ff5252",  # goal
    "#3a3a4a",  # fog of war
])

# --- Config -------------------------------------------------------------------
WIDTH          = 31
HEIGHT         = 31
START          = [1, 1]
GOAL           = [29, 29]
MAZE_SEED      = 7
MAZE_DENSITY   = 0.3   # enough loops for alternate routes — agent reroutes, not deadlocked
NUM_OBSTACLES  = 10
PATROL_TICKS   = [3, 5, 7]
SENSOR_RADIUS  = 7
FRAME_MS       = 200
USE_DSTAR      = True

# ------------------------------------------------------------------------------

def find_side_neighbor(env, path_cell, path_cells_set):
    """Return one free cell adjacent to path_cell that is NOT on the path, or None."""
    cx, cy = path_cell
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = cx + dx, cy + dy
        if env.isValid(nx, ny) and (nx, ny) not in path_cells_set:
            return [nx, ny]
    return None


def scan_sensor(env, agent_pos, known_cells):
    """
    Reveal all cells within SENSOR_RADIUS Manhattan distance.
    Returns newly discovered obstacle cells so the caller can decide to replan.
    """
    newly_discovered = set()
    ax_pos, ay_pos   = agent_pos
    for delta_x in range(-SENSOR_RADIUS, SENSOR_RADIUS + 1):
        for delta_y in range(-SENSOR_RADIUS, SENSOR_RADIUS + 1):
            if abs(delta_x) + abs(delta_y) <= SENSOR_RADIUS:
                cx, cy = ax_pos + delta_x, ay_pos + delta_y
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    cell = (cx, cy)
                    if cell not in known_cells and env.isObstacle(cx, cy):
                        newly_discovered.add(cell)
                    known_cells.add(cell)
    return newly_discovered


def build_display_grid(env, path, agent_pos, goal, known_cells):
    grid = np.full((HEIGHT, WIDTH), UNKNOWN, dtype=int)

    for cx, cy in known_cells:
        grid[cy][cx] = WALL if env.isObstacle(cx, cy) else FREE

    for ox, oy in env.getDynamicObstaclePositions():
        if (ox, oy) in known_cells:
            grid[oy][ox] = DYNAMIC

    for px, py in path:
        if grid[py][px] == FREE:
            grid[py][px] = PATH

    gx, gy = goal
    grid[gy][gx] = GOAL_CELL   # always visible — agent knows its destination

    grid[agent_pos[1]][agent_pos[0]] = AGENT
    return grid


def update_stats_panel(ax_stats, tick, replans, path_length, nodes_history, paused, reached):
    ax_stats.cla()
    ax_stats.set_facecolor("#0d1117")
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])
    for spine in ax_stats.spines.values():
        spine.set_color("#333344")

    if reached:       status = "GOAL REACHED"
    elif paused:      status = "PAUSED"
    elif path_length == 0: status = "BLOCKED"
    else:             status = "RUNNING"

    avg_nodes = int(sum(nodes_history) / len(nodes_history)) if nodes_history else 0

    ax_stats.text(
        0.1, 0.95,
        f"Planner:  {'D* Lite' if USE_DSTAR else 'A*'}\n"
        f"Status:   {status}\n\n"
        f"Tick:     {tick}\n"
        f"Replans:  {replans}\n"
        f"Path len: {path_length}\n\n"
        f"Avg nodes/replan:\n"
        f"  {avg_nodes}\n\n"
        f"Controls:\n"
        f"  Click = wall\n"
        f"  Space = pause\n"
        f"  R     = reset",
        transform=ax_stats.transAxes,
        color="#e8e8e8", fontsize=9,
        verticalalignment="top", fontfamily="monospace",
    )
    ax_stats.set_title("Stats", color="#e8e8e8", fontsize=10, pad=6)


def setup_simulation():
    env = pathplanning.DynamicGridEnvironment(WIDTH, HEIGHT)
    env.setStart(START[0], START[1])
    env.setGoal(GOAL[0], GOAL[1])
    env.generateLabyrinth(MAZE_DENSITY, MAZE_SEED)

    planner      = env.findPathDStar if USE_DSTAR else env.findPath
    initial_path = planner(START[0], START[1], GOAL[0], GOAL[1])

    if not initial_path:
        raise RuntimeError("No path found — try a different MAZE_SEED.")

    path_len = len(initial_path)

    # Anchor obstacles at 1/4, 1/2, 3/4 of the path — spaced through the maze.
    # Each patrols a corridor segment, acting as a slow vehicle blocking the road.
    path_cells_set = {(p[0], p[1]) for p in initial_path}
    anchors        = [int(path_len * (i + 1) / (NUM_OBSTACLES + 1)) for i in range(NUM_OBSTACLES)]

    for obstacle_index, anchor in enumerate(anchors[:NUM_OBSTACLES]):
        anchor_cell   = initial_path[anchor]
        side_neighbor = find_side_neighbor(env, anchor_cell, path_cells_set)
        if side_neighbor:
            if obstacle_index == 0:
                # First obstacle stays blocking until agent reaches it — guaranteed hit.
                ticks = anchor + 5
            else:
                ticks = PATROL_TICKS[obstacle_index % len(PATROL_TICKS)]
            env.addDynamicObstacle([anchor_cell, side_neighbor], ticks)

    if env.getDynamicObstacleCount() == 0:
        raise RuntimeError("Could not place any obstacles.")

    return env, initial_path


def main():
    try:
        env, initial_path = setup_simulation()
    except RuntimeError as error:
        print(error)
        return

    env_ref       = [env]
    agent_pos     = list(START)
    path          = list(initial_path)
    known_cells   = set()
    replans       = [0]
    nodes_history = []
    reached       = [False]
    paused        = [False]

    scan_sensor(env_ref[0], agent_pos, known_cells)
    goal = env_ref[0].getGoal()

    def replan():
        planner  = env_ref[0].findPathDStar if USE_DSTAR else env_ref[0].findPath
        new_path = planner(agent_pos[0], agent_pos[1], goal[0], goal[1])
        path.clear()
        path.extend(new_path)
        replans[0] += 1
        if USE_DSTAR:
            nodes_history.append(env_ref[0].getNodesExploredLastReplan())

    def reset():
        try:
            new_env, new_path = setup_simulation()
        except RuntimeError as error:
            print(error)
            return
        env_ref[0] = new_env
        agent_pos[0], agent_pos[1] = START[0], START[1]
        path.clear()
        path.extend(new_path)
        known_cells.clear()
        scan_sensor(env_ref[0], agent_pos, known_cells)
        replans[0]  = 0
        reached[0]  = False
        paused[0]   = False
        nodes_history.clear()

    # --- Figure ---------------------------------------------------------------
    fig, (ax_grid, ax_stats) = plt.subplots(
        1, 2, figsize=(11, 7),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(bottom=0.1)

    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title(
        f"Live Path Planning — {WIDTH}×{HEIGHT}  |  sensor={SENSOR_RADIUS}  |  {'D* Lite' if USE_DSTAR else 'A*'}",
        color="#e8e8e8", fontsize=10, pad=8,
    )

    img = ax_grid.imshow(
        build_display_grid(env_ref[0], path, agent_pos, goal, known_cells),
        cmap=COLORMAP, vmin=0, vmax=6, interpolation="nearest",
    )

    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
                      ["#e8e8e8", "#1a1a2e", "#ff6600", "#4fc3f7", "#00e676", "#ff5252", "#3a3a4a"]]
    ax_grid.legend(legend_patches,
                   ["Free", "Wall", "Obstacle", "Path", "Agent", "Goal", "Unknown"],
                   loc="lower left", fontsize=7, framealpha=0.85)

    ax_slider    = fig.add_axes([0.35, 0.02, 0.3, 0.025])
    speed_slider = Slider(ax_slider, "ms/frame", 50, 800, valinit=FRAME_MS, valstep=50)

    def on_click(event):
        if event.inaxes != ax_grid or event.xdata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= col < WIDTH and 0 <= row < HEIGHT):
            return
        if env_ref[0].isObstacle(col, row):
            env_ref[0].clearObstacle(col, row)
        else:
            env_ref[0].setObstacle(col, row)
        replan()

    def on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]
        elif event.key == "r":
            reset()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event",    on_key)

    def update(frame):
        try:
            if paused[0] or reached[0]:
                update_stats_panel(ax_stats, env_ref[0].getTickCount(), replans[0],
                                   len(path), nodes_history, paused[0], reached[0])
                return

            env_ref[0].tick()

            scan_sensor(env_ref[0], agent_pos, known_cells)
            next_blocked = len(path) > 1 and env_ref[0].isObstacle(path[1][0], path[1][1])

            if next_blocked or not path:
                replan()

            if len(path) > 1:
                path.pop(0)
                agent_pos[0] = path[0][0]
                agent_pos[1] = path[0][1]

            if agent_pos[0] == goal[0] and agent_pos[1] == goal[1]:
                reached[0] = True

            img.set_data(build_display_grid(env_ref[0], path, agent_pos, goal, known_cells))
            update_stats_panel(ax_stats, env_ref[0].getTickCount(), replans[0],
                               len(path), nodes_history, paused[0], reached[0])

        except Exception as exc:
            print(f"frame error: {exc}")
        fig.canvas.flush_events()

    anim = animation.FuncAnimation(
        fig, update, interval=FRAME_MS, blit=False, cache_frame_data=False
    )
    speed_slider.on_changed(lambda val: setattr(anim.event_source, 'interval', int(val)))

    _ = anim
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
