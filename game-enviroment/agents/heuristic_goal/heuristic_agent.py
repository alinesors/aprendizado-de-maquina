"""
heuristic_agent.py — Agente de Busca Heuristica (A* Visual) para Odisseia Orbital.

Arquitetura modular:
  agents/grid_map.py      — grid 40x30 com margem proporcional a massa
  agents/astar_planner.py — A* geradora com custo gravitacional
  agents/auto_pilot.py    — piloto com velocidade-alvo e evasao
  agents/visualization.py — overlay A*, linha guia e HUD
  agents/replay_buffer.py — save/load de experiencias de sucesso

  Modos:
  python heuristic_agent.py                  → normal (A* + AutoPilot, salva ao vencer)
  python heuristic_agent.py --show 001       → carrega run_001, pula A*
  python heuristic_agent.py --show latest    → carrega o run mais recente
"""

import sys
import os
import math

import pygame

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import config as cfg
from orbital_env import OrbitalEnv

from .grid_map import GridMap
from .astar_planner import AStarPlanner
from .auto_pilot import AutoPilot
from .visualization import (
    simplify_path,
    capture_game_snapshot,
    draw_a_star_overlay,
    draw_web_lines,
    draw_grid_overlay,
    draw_blocked_exclamations,
    draw_current_node_ripple,
    draw_start_beacon,
    draw_path_line,
    draw_agent_hud,
    get_current_goal,
)
from .replay_buffer import save_run, load_run, list_runs

NODES_PER_FRAME = 1
REPLAN_INTERVAL_MS = 1000
PATH_HOLD_FRAMES = 90
PLANNING_FRAME_DELAY = 200


# ============================================================
# Modo Normal — A* ao vivo + salvamento ao vencer
# ============================================================
def main():
    env = OrbitalEnv(render_mode="human")
    obs = env.reset()

    global_margin = 0.0
    attempt = 1
    crash_info = ""
    crash_timer = 0
    crash_handled = False

    total_thrusts = 0
    action_log = []
    checkpoint_action_indices = []
    segments = []
    checkpoint_order = []
    current_segment_target = None
    current_segment_path = None
    current_segment_raw = None

    grid_map = GridMap(env.planets, env.checkpoints, env.station_pos,
                       global_margin)
    ship_pos = (float(obs[0]), float(obs[1]))
    goal_pos = get_current_goal(env, ship_pos)
    current_segment_target = goal_pos

    planner = AStarPlanner(grid_map)
    a_star_gen = planner.search(ship_pos, goal_pos)
    autopilot = None
    path = None

    phase = "planning"
    last_replan_time = pygame.time.get_ticks()
    last_dist_to_goal = None

    planning_background = None
    start_cell = grid_map.pos_to_cell(*ship_pos)
    path_hold_counter = 0
    last_search_state = None

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    status = env.last_info.get("status", "") if env.done else ""
                    if not env.done or status == "docked":
                        global_margin += 5.0
                    attempt += 1
                    crash_info = (
                        f"TENTATIVA {attempt}  MARGEM +{global_margin:.0f}px"
                    )
                    crash_timer = 120
                    crash_handled = False
                    total_thrusts = 0
                    action_log = []
                    checkpoint_action_indices = []
                    segments = []
                    checkpoint_order = []
                    current_segment_path = None
                    current_segment_raw = None

                    obs = env.reset()
                    grid_map = GridMap(env.planets, env.checkpoints,
                                       env.station_pos, global_margin)
                    ship_pos = (float(obs[0]), float(obs[1]))
                    goal_pos = get_current_goal(env, ship_pos)
                    current_segment_target = goal_pos

                    planner = AStarPlanner(grid_map)
                    a_star_gen = planner.search(ship_pos, goal_pos)
                    autopilot = None
                    path = None
                    phase = "planning"
                    last_replan_time = pygame.time.get_ticks()
                    last_dist_to_goal = None
                    planning_background = None
                    start_cell = grid_map.pos_to_cell(*ship_pos)
                    path_hold_counter = 0

        if phase == "planning":
            if planning_background is None:
                env.render()
                planning_background = capture_game_snapshot(env)
            else:
                env.screen.blit(planning_background, (0, 0))

            search_state = None

            if path_hold_counter > 0:
                path_hold_counter -= 1

                if last_search_state is not None:
                    draw_grid_overlay(env.screen)
                    draw_a_star_overlay(env.screen, last_search_state)
                    draw_web_lines(env.screen, last_search_state.get("came_from", {}))
                    draw_start_beacon(env.screen, start_cell)
                    draw_blocked_exclamations(
                        env.screen,
                        last_search_state.get("blocked", []),
                        env.planets,
                    )

                if current_segment_path:
                    draw_path_line(env.screen, current_segment_path)

                if path_hold_counter == 0:
                    phase = "flying"
                    last_replan_time = pygame.time.get_ticks()
                    last_dist_to_goal = math.hypot(
                        obs[0] - goal_pos[0], obs[1] - goal_pos[1]
                    )
            else:
                for _ in range(NODES_PER_FRAME):
                    try:
                        search_state = next(a_star_gen)
                    except StopIteration:
                        break
                    if search_state.get("path") is not None:
                        break

            if search_state is not None:
                last_search_state = search_state
                draw_grid_overlay(env.screen)
                draw_a_star_overlay(env.screen, search_state)
                draw_web_lines(env.screen, search_state.get("came_from", {}))
                draw_current_node_ripple(
                    env.screen, search_state.get("current")
                )
                draw_start_beacon(env.screen, start_cell)
                draw_blocked_exclamations(
                    env.screen,
                    search_state.get("blocked", []),
                    env.planets,
                )

                found_path = search_state.get("path")
                if found_path is not None:
                    a_star_gen = None

                    uncollected = [cp for cp in env.checkpoints if not cp["collected"]]
                    is_dock = len(uncollected) == 0

                    if found_path:
                        current_segment_raw = list(found_path)
                        path = simplify_path(found_path)
                        autopilot = AutoPilot(path, env.planets, is_docking=is_dock)
                    else:
                        current_segment_raw = [ship_pos, goal_pos]
                        path = [ship_pos, goal_pos]
                        autopilot = AutoPilot(path, env.planets, is_docking=is_dock)

                    current_segment_path = path

                    draw_path_line(env.screen, path)
                    path_hold_counter = PATH_HOLD_FRAMES
                else:
                    pygame.time.wait(PLANNING_FRAME_DELAY)

            if crash_timer > 0:
                draw_agent_hud(env.screen, attempt, global_margin,
                               crash_info, crash_timer)
                crash_timer -= 1
                if crash_timer == 0 and env.done:
                    status = env.last_info.get("status", "")
                    if status != "docked":
                        obs = env.reset()
                        crash_handled = False
                        total_thrusts = 0
                        action_log = []
                        checkpoint_action_indices = []
                        segments = []
                        checkpoint_order = []
                        current_segment_path = None
                        current_segment_raw = None
                        grid_map = GridMap(env.planets, env.checkpoints,
                                           env.station_pos, global_margin)
                        ship_pos = (float(obs[0]), float(obs[1]))
                        goal_pos = get_current_goal(env, ship_pos)
                        current_segment_target = goal_pos
                        planner = AStarPlanner(grid_map)
                        a_star_gen = planner.search(ship_pos, goal_pos)
                        autopilot = None
                        path = None
                        phase = "planning"
                        last_replan_time = pygame.time.get_ticks()
                        last_dist_to_goal = None
                        planning_background = None
                        start_cell = grid_map.pos_to_cell(*ship_pos)
                        path_hold_counter = 0

            pygame.display.flip()

        elif phase == "flying":
            if not env.done:
                action = autopilot.get_action(obs) if autopilot else 0
                action_log.append(action)
                obs, reward, done, info = env.step(action)
                if action != 0:
                    total_thrusts += 1

            env.render()

            if path:
                draw_path_line(env.screen, path)

            if crash_timer > 0:
                draw_agent_hud(env.screen, attempt, global_margin,
                               crash_info, crash_timer)
                crash_timer -= 1

            pygame.display.flip()

            if env.done:
                if not crash_handled:
                    status = env.last_info.get("status", "")
                    if status == "docked":
                        if current_segment_target is not None:
                            segments.append({
                                "target": current_segment_target,
                                "waypoints": current_segment_raw
                                if current_segment_raw
                                else [],
                                "display_waypoints": current_segment_path
                                if current_segment_path
                                else [],
                            })

                        stats = {
                            "steps": env.episode_steps,
                            "thrusts": total_thrusts,
                            "fuel_used": (
                                cfg.MAX_FUEL - float(obs[4])
                            ),
                            "reward": float(env.episode_reward),
                            "checkpoints": sum(
                                1 for cp in env.checkpoints
                                if cp["collected"]
                            ),
                        }
                        run_id, filepath = save_run(
                            segments, checkpoint_order, stats, action_log,
                            checkpoint_action_indices,
                        )
                        crash_info = (
                            f"SUCESSO!  Salvo run_{run_id:03d}.json"
                        )
                        crash_timer = 300
                    else:
                        global_margin += 5.0
                        attempt += 1
                        crash_info = (
                            f"TENTATIVA {attempt}"
                            f"  MARGEM +{global_margin:.0f}px"
                        )
                        crash_timer = 120
                    crash_handled = True
            else:
                crash_handled = False

                if info.get("checkpoint_collected"):
                    checkpoint_action_indices.append(len(action_log))
                    if current_segment_target is not None:
                        segments.append({
                            "target": current_segment_target,
                            "waypoints": current_segment_raw
                            if current_segment_raw
                            else [],
                        })
                        checkpoint_order.append(current_segment_target)

                    grid_map = GridMap(env.planets, env.checkpoints,
                                       env.station_pos, global_margin)
                    ship_pos = (float(obs[0]), float(obs[1]))
                    goal_pos = get_current_goal(env, ship_pos)
                    current_segment_target = goal_pos
                    current_segment_path = None
                    current_segment_raw = None

                    planner = AStarPlanner(grid_map)
                    a_star_gen = planner.search(ship_pos, goal_pos)
                    path = None
                    autopilot = None
                    phase = "planning"
                    last_replan_time = pygame.time.get_ticks()
                    last_dist_to_goal = None
                    planning_background = None
                    start_cell = grid_map.pos_to_cell(*ship_pos)
                    path_hold_counter = 0
                    continue

                now = pygame.time.get_ticks()
                if (
                    now - last_replan_time > REPLAN_INTERVAL_MS
                    and current_segment_target is not None
                ):
                    current_dist = math.hypot(
                        obs[0] - current_segment_target[0],
                        obs[1] - current_segment_target[1],
                    )
                    if (
                        last_dist_to_goal is not None
                        and current_dist > last_dist_to_goal + 10
                    ):
                        if current_segment_target is not None:
                            # CORREÇÃO: Salva o momento exato em que a rota foi recalculada!
                            checkpoint_action_indices.append(len(action_log))
                            
                            segments.append({
                                "target": current_segment_target,
                                "waypoints": current_segment_raw
                                if current_segment_raw
                                else [],
                                "display_waypoints": current_segment_path
                                if current_segment_path
                                else [],
                            })

                        grid_map = GridMap(env.planets, env.checkpoints,
                                           env.station_pos, global_margin)
                        ship_pos = (float(obs[0]), float(obs[1]))
                        goal_pos = get_current_goal(env, ship_pos)
                        current_segment_target = goal_pos
                        current_segment_path = None
                        current_segment_raw = None

                        planner = AStarPlanner(grid_map)
                        a_star_gen = planner.search(ship_pos, goal_pos)
                        path = None
                        autopilot = None
                        phase = "planning"
                        last_replan_time = now
                        last_dist_to_goal = None
                        planning_background = None
                        start_cell = grid_map.pos_to_cell(*ship_pos)
                        path_hold_counter = 0
                        continue
                    last_dist_to_goal = current_dist
                    last_replan_time = now

    env.close()
    pygame.quit()
    sys.exit()


# ============================================================
# Modo Treinado — replay deterministico de acoes + fallback
# ============================================================
def main_trained(run_id):
    run_data = load_run(run_id)
    if run_data is None:
        print(f"Run '{run_id}' nao encontrado.")
        print("Runs disponiveis:")
        for r in list_runs():
            rid = r["id"]
            rts = r["timestamp"][:19] if r["timestamp"] else "?"
            print(f"  run_{rid:03d}  {rts}  reward={r['reward']:.0f}")
        sys.exit(1)

    env = OrbitalEnv(render_mode="human")
    obs = env.reset()

    actions = run_data.get("actions", [])
    segments = run_data["segments"]
    stats = run_data["stats"]
    checkpoint_action_indices = run_data.get(
        "checkpoint_action_indices", []
    )
    has_actions = len(actions) > 0

    print(f"Run {run_data['run_id']:03d} carregado:")
    if has_actions:
        print(f"  Acoes: {len(actions)}  (replay deterministico)")
    else:
        print(f"  Acoes: 0  (fallback: AutoPilot com waypoints)")
    print(f"  Segmentos: {len(segments)}")
    print(f"  Steps: {stats['steps']}  Thrusts: {stats['thrusts']}")
    print(f"  Reward: {stats['reward']:.0f}")

    action_idx = 0
    step_counter = 0
    collected_count = 0
    seg_idx = 0
    crash_timer = 0
    crash_info = ""
    display_path = None
    autopilot = None
    info_printed = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    action_idx = 0
                    step_counter = 0
                    collected_count = 0
                    seg_idx = 0
                    display_path = None
                    autopilot = None
                    info_printed = False

        if env.done:
            env.render()
            if display_path:
                draw_path_line(env.screen, display_path)
            if not info_printed:
                status = env.last_info.get("status", "")
                icon = "ATRACOU" if status == "docked" else (
                    "COLIDIU" if status == "crashed_planet" else (
                        "FORA" if status == "out_of_bounds" else (
                            "SEM COMB." if status == "no_fuel" else "???"
                        )
                    )
                )
                print(f"Run {run_data['run_id']:03d} | {icon} | steps={step_counter} | CPs={collected_count}")
                info_printed = True
            pygame.display.flip()
            continue

        if not env.done and step_counter >= stats["steps"] and has_actions:
            # Acoes do replay acabaram — talvez precise do fallback autopilot
            # Deixa continuar com autopilot ou para
            pass

        if has_actions:
            if action_idx < len(actions):
                action = actions[action_idx]
                action_idx += 1
            else:
                if autopilot is None:
                    seg = segments[seg_idx] if seg_idx < len(segments) else segments[-1]
                    if seg.get("waypoints"):
                        path = [list(wp) for wp in seg["waypoints"]]
                        autopilot = AutoPilot(path, env.planets)
                    else:
                        autopilot = None
                if autopilot is None or not autopilot.has_path():
                    seg_idx += 1
                    autopilot = None
                    if seg_idx < len(segments):
                        seg = segments[seg_idx]
                        if seg.get("waypoints"):
                            path = [list(wp) for wp in seg["waypoints"]]
                            autopilot = AutoPilot(path, env.planets)
                action = autopilot.get_action(obs) if autopilot else 0
        else:
            if autopilot is None:
                if seg_idx < len(segments):
                    seg = segments[seg_idx]
                    path = [list(wp) for wp in seg.get("waypoints", [])]
                    if path:
                        autopilot = AutoPilot(path, env.planets)
                else:
                    autopilot = None
            if autopilot is None:
                action = 0
            elif not autopilot.has_path():
                seg_idx += 1
                autopilot = None
                action = 0
            else:
                action = autopilot.get_action(obs)

        obs, reward, done, info = env.step(action)
        step_counter += 1

        if checkpoint_action_indices:
            while seg_idx < len(checkpoint_action_indices):
                if step_counter >= checkpoint_action_indices[seg_idx]:
                    seg_idx += 1
                else:
                    break

        if seg_idx < len(segments):
            seg = segments[seg_idx]
            display_path = [list(wp)
                            for wp in seg.get("display_waypoints",
                                              seg["waypoints"])]

        env.render()
        if display_path:
            draw_path_line(env.screen, display_path)

        if env.done:
            continue
        elif info.get("checkpoint_collected"):
            collected_count += 1
            if not checkpoint_action_indices:
                seg_idx = max(seg_idx,
                              min(collected_count, len(segments) - 1))
            autopilot = None

        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit()


# ============================================================
# Entrada
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] in ("--trained", "--show"):
        main_trained(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "--list":
        runs = list_runs()
        if runs:
            print("Runs salvos:")
            for r in runs:
                ts = r["timestamp"][:19] if r["timestamp"] else "?"
                print(
                    f"  run_{r['id']:03d}  {ts}"
                    f"  steps={r['steps']}  thrusts={r['thrusts']}"
                    f"  fuel={r['fuel_used']:.0f}  reward={r['reward']:.0f}"
                )
        else:
            print("Nenhum run salvo ainda.")
    else:
        main()
