# python/live_demo.py
# Live simulation: agent navigates a maze with fog-of-war and patrolling obstacles.
# D* Lite or A* replanning, multi-agent, random walkers, visual polish.
#
# Run: source venv/bin/activate && python python/live_demo.py
# Controls: click=wall | space=pause | R=reset | D=toggle planner | slider=speed

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import pathplanning

# --- Display values -----------------------------------------------------------
FREE         = 0
WALL         = 1
DYNAMIC      = 2
PATH         = 3
AGENT        = 4
GOAL_CELL    = 5
UNKNOWN      = 6
BREADCRUMB   = 7
PATROL_ZONE  = 8
PATH_REPLAN  = 9
AGENT2       = 10

COLORMAP = mcolors.ListedColormap([
    "#e8e8e8",  # free
    "#1a1a2e",  # wall
    "#ff6600",  # dynamic obstacle
    "#4fc3f7",  # planned path
    "#00e676",  # agent 1
    "#ff5252",  # goal
    "#3a3a4a",  # fog of war
    "#6aaa6a",  # breadcrumb trail
    "#3a2510",  # patrol zone hint
    "#ff2020",  # path flash on replan
    "#ffe000",  # agent 2
])

# --- Config -------------------------------------------------------------------
WIDTH            = 31
HEIGHT           = 31
START            = [1, 1]
GOAL             = [29, 29]
# Secondary agents — one per remaining corner, each navigates to the opposite corner
SECONDARY_AGENTS = [
    {"start": [29, 29], "goal": [1,  1]},
    {"start": [1,  29], "goal": [29, 1]},
    {"start": [29,  1], "goal": [1, 29]},
]
MAZE_SEED        = 7
MAZE_DENSITY     = 0.3
NUM_OBSTACLES    = 5
NUM_RANDOM_OBS   = 8
PATROL_TICKS     = [3, 5, 7]
SENSOR_RADIUS    = 7
BREADCRUMB_LEN   = 25
FLASH_DURATION   = 6         # frames path flashes red after replan
FRAME_MS         = 200

# ------------------------------------------------------------------------------

def find_side_neighbor(env, path_cell, path_cells_set):
    cx, cy = path_cell
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = cx + dx, cy + dy
        if env.isValid(nx, ny) and (nx, ny) not in path_cells_set:
            return [nx, ny]
    return None


def scan_sensor(env, agent_pos, known_cells):
    ax_pos, ay_pos = agent_pos
    for delta_x in range(-SENSOR_RADIUS, SENSOR_RADIUS + 1):
        for delta_y in range(-SENSOR_RADIUS, SENSOR_RADIUS + 1):
            if abs(delta_x) + abs(delta_y) <= SENSOR_RADIUS:
                cx, cy = ax_pos + delta_x, ay_pos + delta_y
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    known_cells.add((cx, cy))


def step_random_obstacles(env, random_obs):
    """Move each random obstacle to a random free adjacent cell."""
    for obs in random_obs:
        cx, cy = obs
        neighbors = [
            [cx + dx, cy + dy]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if env.isValid(cx + dx, cy + dy)
        ]
        if neighbors:
            env.clearObstacle(cx, cy)
            chosen = random.choice(neighbors)
            env.setObstacle(chosen[0], chosen[1])
            obs[0], obs[1] = chosen


def build_display_grid(env, path, agent_pos, secondary_agents, goal, known_cells,
                       breadcrumbs, patrol_zones, flash_ticks, random_obs):
    grid = np.full((HEIGHT, WIDTH), UNKNOWN, dtype=int)

    for cx, cy in known_cells:
        grid[cy][cx] = WALL if env.isObstacle(cx, cy) else FREE

    for cx, cy in patrol_zones:
        if (cx, cy) in known_cells and grid[cy][cx] == FREE:
            grid[cy][cx] = PATROL_ZONE

    for cx, cy in breadcrumbs:
        if grid[cy][cx] in (FREE, PATROL_ZONE):
            grid[cy][cx] = BREADCRUMB

    for ox, oy in env.getDynamicObstaclePositions():
        if (ox, oy) in known_cells:
            grid[oy][ox] = DYNAMIC

    for rx, ry in random_obs:
        if (rx, ry) in known_cells:
            grid[ry][rx] = DYNAMIC

    path_color = PATH_REPLAN if flash_ticks > 0 else PATH
    for px, py in path:
        if grid[py][px] in (FREE, PATROL_ZONE, BREADCRUMB):
            grid[py][px] = path_color

    gx, gy = goal
    grid[gy][gx] = GOAL_CELL

    for agent in secondary_agents:
        ax2, ay2 = agent["pos"]
        grid[ay2][ax2] = AGENT2

    grid[agent_pos[1]][agent_pos[0]] = AGENT
    return grid


def update_stats_panel(ax_stats, tick, replans, replans2, path_length,
                       nodes_history, paused, reached, use_dstar):
    ax_stats.cla()
    ax_stats.set_facecolor("#0d1117")
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])
    for spine in ax_stats.spines.values():
        spine.set_color("#333344")

    if reached:            status = "GOAL REACHED"
    elif paused:           status = "PAUSED"
    elif path_length == 0: status = "BLOCKED"
    else:                  status = "RUNNING"

    avg_nodes = int(sum(nodes_history) / len(nodes_history)) if nodes_history else 0

    ax_stats.text(
        0.1, 0.95,
        f"Planner:  {'D* Lite' if use_dstar else 'A*'}\n"
        f"Status:   {status}\n\n"
        f"Tick:     {tick}\n"
        f"Replans:  {replans}\n"
        f"Agent2 replans: {replans2}\n"
        f"Path len: {path_length}\n\n"
        f"Avg nodes/replan:\n"
        f"  {avg_nodes}\n\n"
        f"Controls:\n"
        f"  Click = wall\n"
        f"  Space = pause\n"
        f"  R     = reset\n"
        f"  D     = planner",
        transform=ax_stats.transAxes,
        color="#e8e8e8", fontsize=9,
        verticalalignment="top", fontfamily="monospace",
    )
    ax_stats.set_title("Stats", color="#e8e8e8", fontsize=10, pad=6)


def setup_simulation(use_dstar):
    env = pathplanning.DynamicGridEnvironment(WIDTH, HEIGHT)
    env.setStart(START[0], START[1])
    env.setGoal(GOAL[0], GOAL[1])
    env.generateLabyrinth(MAZE_DENSITY, MAZE_SEED)

    planner      = env.findPathDStar if use_dstar else env.findPath
    initial_path = planner(START[0], START[1], GOAL[0], GOAL[1])
    if not initial_path:
        raise RuntimeError("No path found — try a different MAZE_SEED.")

    path_len       = len(initial_path)
    path_cells_set = {(p[0], p[1]) for p in initial_path}
    patrol_zones   = set()
    anchors        = [int(path_len * (i + 1) / (NUM_OBSTACLES + 1)) for i in range(NUM_OBSTACLES)]

    for obstacle_index, anchor in enumerate(anchors):
        anchor_cell   = initial_path[anchor]
        side_neighbor = find_side_neighbor(env, anchor_cell, path_cells_set)
        if side_neighbor:
            ticks = (anchor + 5) if obstacle_index == 0 else PATROL_TICKS[obstacle_index % len(PATROL_TICKS)]
            env.addDynamicObstacle([anchor_cell, side_neighbor], ticks)
            patrol_zones.add((anchor_cell[0], anchor_cell[1]))
            patrol_zones.add((side_neighbor[0], side_neighbor[1]))

    if env.getDynamicObstacleCount() == 0:
        raise RuntimeError("Could not place any obstacles.")

    # Place random-walk obstacles in free cells not on the path.
    random_obs = []
    candidates = [
        [x, y] for x in range(1, WIDTH - 1) for y in range(1, HEIGHT - 1)
        if not env.isObstacle(x, y) and (x, y) not in path_cells_set
        and abs(x - START[0]) + abs(y - START[1]) > 5
        and abs(x - GOAL[0])  + abs(y - GOAL[1])  > 5
    ]
    random.shuffle(candidates)
    for cell in candidates[:NUM_RANDOM_OBS]:
        env.setObstacle(cell[0], cell[1])
        random_obs.append(cell)

    return env, initial_path, patrol_zones, random_obs


def main():
    use_dstar = [True]

    try:
        env, initial_path, patrol_zones, random_obs = setup_simulation(use_dstar[0])
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
    stuck_ticks   = [0]
    pos_history   = [list(START)]
    breadcrumbs   = []
    flash_ticks   = [0]
    patrol_ref    = [patrol_zones]
    random_ref    = [random_obs]

    # Secondary agents — one dict per agent with all mutable state
    secondary = [
        {"pos": list(cfg["start"]), "goal": list(cfg["goal"]),
         "path": [], "known": set(), "reached": False, "replans": 0}
        for cfg in SECONDARY_AGENTS
    ]

    scan_sensor(env_ref[0], agent_pos, known_cells)
    for agent in secondary:
        scan_sensor(env_ref[0], agent["pos"], agent["known"])
    goal = env_ref[0].getGoal()

    def replan():
        for agent in secondary:
            env_ref[0].setObstacle(agent["pos"][0], agent["pos"][1])
        planner  = env_ref[0].findPathDStar if use_dstar[0] else env_ref[0].findPath
        new_path = planner(agent_pos[0], agent_pos[1], goal[0], goal[1])
        for agent in secondary:
            env_ref[0].clearObstacle(agent["pos"][0], agent["pos"][1])
        path.clear()
        path.extend(new_path)
        replans[0] += 1
        flash_ticks[0] = FLASH_DURATION
        if use_dstar[0]:
            nodes_history.append(env_ref[0].getNodesExploredLastReplan())

    def replan_secondary(agent):
        if agent["reached"]:
            return
        env_ref[0].setObstacle(agent_pos[0], agent_pos[1])
        new_path = env_ref[0].findPath(agent["pos"][0], agent["pos"][1],
                                       agent["goal"][0], agent["goal"][1])
        env_ref[0].clearObstacle(agent_pos[0], agent_pos[1])
        agent["path"].clear()
        agent["path"].extend(new_path)
        agent["replans"] += 1

    def reset():
        try:
            new_env, new_path, new_zones, new_robs = setup_simulation(use_dstar[0])
        except RuntimeError as error:
            print(error)
            return
        env_ref[0]    = new_env
        patrol_ref[0] = new_zones
        random_ref[0] = new_robs
        agent_pos[0], agent_pos[1] = START[0], START[1]
        path.clear(); path.extend(new_path)
        known_cells.clear(); scan_sensor(env_ref[0], agent_pos, known_cells)
        for i, agent in enumerate(secondary):
            cfg = SECONDARY_AGENTS[i]
            agent["pos"]     = list(cfg["start"])
            agent["goal"]    = list(cfg["goal"])
            agent["path"]    = []
            agent["known"]   = set()
            agent["reached"] = False
            agent["replans"] = 0
            scan_sensor(env_ref[0], agent["pos"], agent["known"])
        replans[0]  = 0
        reached[0]  = False
        paused[0]   = False
        stuck_ticks[0] = 0
        pos_history.clear(); pos_history.append(list(START))
        breadcrumbs.clear()
        flash_ticks[0] = 0
        nodes_history.clear()

    for agent in secondary:
        replan_secondary(agent)

    # --- Figure ---------------------------------------------------------------
    fig, (ax_grid, ax_stats) = plt.subplots(
        1, 2, figsize=(11, 7),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(bottom=0.1)

    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    def grid_title():
        ax_grid.set_title(
            f"Live Path Planning — {WIDTH}×{HEIGHT}  |  sensor={SENSOR_RADIUS}  |  {'D* Lite' if use_dstar[0] else 'A*'}",
            color="#e8e8e8", fontsize=10, pad=8,
        )

    grid_title()

    img = ax_grid.imshow(
        build_display_grid(env_ref[0], path, agent_pos, secondary, goal,
                           known_cells, breadcrumbs, patrol_ref[0], flash_ticks[0], random_ref[0]),
        cmap=COLORMAP, vmin=0, vmax=10, interpolation="nearest",
    )

    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
                      ["#e8e8e8","#1a1a2e","#ff6600","#4fc3f7","#ff2020",
                       "#00e676","#ffe000","#ff5252","#6aaa6a","#3a2510","#3a3a4a"]]
    ax_grid.legend(legend_patches,
                   ["Free","Wall","Obstacle","Path","Replan","Agent1","Agent2",
                    "Goal","Trail","Patrol zone","Unknown"],
                   loc="lower left", fontsize=6, framealpha=0.85, ncol=2)

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
        img.set_data(build_display_grid(env_ref[0], path, agent_pos, secondary, goal,
                                        known_cells, breadcrumbs, patrol_ref[0], flash_ticks[0], random_ref[0]))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]
        elif event.key in ("r", "R"):
            reset()
            grid_title()
            img.set_data(build_display_grid(env_ref[0], path, agent_pos, secondary, goal,
                                            known_cells, breadcrumbs, patrol_ref[0], flash_ticks[0], random_ref[0]))
            fig.canvas.draw_idle()
        elif event.key in ("d", "D"):
            use_dstar[0] = not use_dstar[0]
            grid_title()
            replan()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event",    on_key)

    def update(frame):
        try:
            if paused[0] or reached[0]:
                update_stats_panel(ax_stats, env_ref[0].getTickCount(), replans[0], replans2[0],
                                   len(path), nodes_history, paused[0], reached[0], use_dstar[0])
                return

            env_ref[0].tick()
            step_random_obstacles(env_ref[0], random_ref[0])

            scan_sensor(env_ref[0], agent_pos,  known_cells)
            for agent in secondary:
                scan_sensor(env_ref[0], agent["pos"], agent["known"])

            if flash_ticks[0] > 0:
                flash_ticks[0] -= 1

            # --- Agent 1 ------------------------------------------------------
            next_blocked = len(path) > 1 and env_ref[0].isObstacle(path[1][0], path[1][1])
            if next_blocked or not path:
                replan()

            if len(path) > 1:
                path.pop(0)
                agent_pos[0] = path[0][0]
                agent_pos[1] = path[0][1]
                breadcrumbs.append((agent_pos[0], agent_pos[1]))
                if len(breadcrumbs) > BREADCRUMB_LEN:
                    breadcrumbs.pop(0)
                pos_history.append(list(agent_pos))
                if len(pos_history) > 60:
                    pos_history.pop(0)
                stuck_ticks[0] = 0
            else:
                stuck_ticks[0] += 1

            if stuck_ticks[0] > 3 and len(pos_history) > 1:
                pos_history.pop()
                prev = pos_history[-1]
                if not env_ref[0].isObstacle(prev[0], prev[1]):
                    agent_pos[0] = prev[0]
                    agent_pos[1] = prev[1]
                    replan()

            if agent_pos[0] == goal[0] and agent_pos[1] == goal[1]:
                reached[0] = True

            # --- Secondary agents ---------------------------------------------
            for agent in secondary:
                if agent["reached"]:
                    continue
                agent_path = agent["path"]
                scan_sensor(env_ref[0], agent["pos"], agent["known"])
                next_blocked2 = len(agent_path) > 1 and env_ref[0].isObstacle(
                    agent_path[1][0], agent_path[1][1])
                if next_blocked2 or not agent_path:
                    replan_secondary(agent)
                if len(agent_path) > 1:
                    agent_path.pop(0)
                    agent["pos"][0] = agent_path[0][0]
                    agent["pos"][1] = agent_path[0][1]
                if agent["pos"] == agent["goal"]:
                    agent["reached"] = True

            img.set_data(build_display_grid(env_ref[0], path, agent_pos, secondary, goal,
                                            known_cells, breadcrumbs, patrol_ref[0], flash_ticks[0], random_ref[0]))
            update_stats_panel(ax_stats, env_ref[0].getTickCount(), replans[0],
                               sum(a["replans"] for a in secondary),
                               len(path), nodes_history, paused[0], reached[0], use_dstar[0])

        except Exception as exc:
            print(f"frame error: {exc}")
        fig.canvas.flush_events()

    anim = animation.FuncAnimation(
        fig, update, interval=FRAME_MS, blit=False, cache_frame_data=False
    )

    def on_speed_change(val):
        anim.event_source.interval = int(val)

    speed_slider.on_changed(on_speed_change)

    _ = anim
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
