import sys
import os
import math
import numpy as np

_project_dir = os.path.dirname(os.path.abspath(__file__))
_game_env = os.path.join(_project_dir, "game-enviroment")
if _game_env not in sys.path:
    sys.path.insert(0, _game_env)

import pygame
import config as cfg
from orbital_env import OrbitalEnv
from agents.genetic.genetic_env import AmbienteGenetico
from agents.genetic.treinador import TreinadorGenetico
from agents.genetic.ship import CerebroNave

MAX_STEPS = 2000
AUTO_RESET_FRAMES = 90


def obter_sensores_orbital(env, obs):
    pos = np.array([obs[0], obs[1]], dtype=np.float64)
    vel = np.array([obs[2], obs[3]], dtype=np.float64)
    fuel = float(obs[4])

    best_dist = float('inf')
    target_pos = env.station_pos
    for cp in env.checkpoints:
        if not cp["collected"]:
            d = float(np.linalg.norm(cp["pos"] - pos))
            if d < best_dist:
                best_dist = d
                target_pos = cp["pos"]

    dx = float(target_pos[0] - pos[0])
    dy = float(target_pos[1] - pos[1])
    vx = float(vel[0])
    vy = float(vel[1])

    launch_planet = env.planets[cfg.LAUNCH_PLANET_INDEX]
    dist_to_launch = float(np.linalg.norm(pos - launch_planet["pos"]))
    escaped = dist_to_launch > cfg.LAUNCH_ESCAPE_DISTANCE

    perigos = []
    for i, planet in enumerate(env.planets):
        if not escaped and i == cfg.LAUNCH_PLANET_INDEX:
            continue
        dist = float(np.linalg.norm(pos - planet["pos"]))
        surface_dist = dist - planet["radius"]
        p_dx = float(planet["pos"][0] - pos[0])
        p_dy = float(planet["pos"][1] - pos[1])
        perigos.append((surface_dist, p_dx, p_dy))

    perigos.sort(key=lambda x: x[0])
    p1 = perigos[0] if len(perigos) > 0 else (999.0, 0.0, 0.0)
    p2 = perigos[1] if len(perigos) > 1 else (999.0, 0.0, 0.0)

    dist_alvo = max(math.hypot(dx, dy), 1.0)
    vel_toward = (vx * dx + vy * dy) / dist_alvo / 2.75

    return [dx, dy, vx, vy, fuel, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], vel_toward]


def modo_debug():
    pop_size = 200
    env = AmbienteGenetico(pop_size=pop_size, render_mode="human")

    running = True
    dead_frames = 0
    fitness_done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    if env.total_alive == 0:
                        dead_frames = AUTO_RESET_FRAMES
                    else:
                        env.reset()
                        env.generation = 1
                        fitness_done = False

        if env.total_alive == 0:
            if not fitness_done:
                env.calcular_fitness_frota()
                top = sorted(env.frota, key=lambda n: n.fitness, reverse=True)[:3]
                print("Geracao {} | {} steps | top3: {}".format(
                    env.generation, env.episode_steps,
                    [round(n.fitness) for n in top]))
                fitness_done = True

            env.render()

            dead_frames += 1
            if dead_frames >= AUTO_RESET_FRAMES:
                env.evoluir_geracao()
                env.reset()
                dead_frames = 0
                fitness_done = False
            continue

        actions = []
        for nave in env.frota:
            if nave.ativa:
                sensores = env.obter_sensores_nave(nave)
                acao = nave.cerebro.prever_acao(sensores)
                actions.append(acao)
            else:
                actions.append(0)

        env.step(actions)
        env.render()

    env.close()


def modo_treino(pop_size, geracoes):
    print("=" * 60)
    print("ODISSEIA ORBITAL — Treinamento Genetico")
    print("=" * 60)
    print("Populacao: {} naves".format(pop_size))
    print("Geracoes:  {}".format(geracoes))
    print()

    treinador = TreinadorGenetico(pop_size=pop_size)
    treinador.treinar(geracoes=geracoes, max_steps=2000,
                      taxa_mutacao=0.06, forca_mutacao=0.15, decay_forca=0.9992,
                      showcase=True)


def modo_show(gen_id):
    data = TreinadorGenetico.carregar(gen_id)
    if data is None:
        print("Checkpoint nao encontrado. Execute --train primeiro.")
        return

    cerebro = CerebroNave()
    cerebro.definir_cromossomo(data["cromossomo"])

    generation = data["generation"]

    print("=" * 50)
    print("  ODISSEIA ORBITAL — Genetico Showcase (Ambiente Padrao)")
    print(f"  Geracao: {generation}  |  Fitness: {data['fitness']:.0f}")
    print("=" * 50)

    env = OrbitalEnv(render_mode="human")
    obs = env.reset()

    running = True
    info_printed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    info_printed = False

        if not env.done and env.episode_steps < MAX_STEPS:
            sensores = obter_sensores_orbital(env, obs)
            acao = cerebro.prever_acao(sensores)
            obs, reward, terminal, info = env.step(acao)

        env.render()

        if env.done and not info_printed:
            status = env.last_info.get("status", "")
            collected = sum(1 for cp in env.checkpoints if cp["collected"])
            icone = "ATRACOU" if status == "docked" else (
                "COLIDIU" if status == "crashed_planet" else (
                    "FORA" if status == "out_of_bounds" else (
                        "SEM COMB." if status == "no_fuel" else "TIMEOUT"
                    )
                )
            )
            print("Showcase gen {} | {} | steps={} | cp={}/{} | fuel={:.0f} | rec={:.0f}".format(
                generation, icone, env.episode_steps,
                collected, len(env.checkpoints), obs[4], env.episode_reward))
            info_printed = True

    env.close()


def modo_listar():
    cps = TreinadorGenetico.listar()
    if cps:
        print("Checkpoints salvos:")
        for c in cps:
            status = "DOCADO" if c["docked"] else "---"
            print("  {}  gen {:3d}  fit={:.0f}  cp={}  {}".format(
                c["file"], c["generation"], c["fitness"],
                c["checkpoints"], status))
    else:
        print("Nenhum checkpoint salvo. Execute com --train primeiro.")


def print_help():
    print("Uso: python run_genetic.py [modo] [opcoes]")
    print()
    print("Modos:")
    print("  (sem argumentos)    Debug visual com evolucao live")
    print("  --train             Treino headless (500 geracoes)")
    print("  --train --eps N     Treino com N geracoes")
    print("  --train --pop N     Treino com N naves")
    print("  --show              Showcase do melhor cerebro (best.pkl)")
    print("  --show N            Showcase de uma geracao especifica")
    print("  --list              Listar checkpoints salvos")


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_help()
        return

    if "--train" in args:
        pop_size = 200
        geracoes = 500
        for i, arg in enumerate(args):
            if arg in ("--eps", "--gens") and i + 1 < len(args):
                geracoes = int(args[i + 1])
            elif arg == "--pop" and i + 1 < len(args):
                pop_size = int(args[i + 1])
        modo_treino(pop_size, geracoes)
        return

    if "--show" in args:
        gen_id = None
        for i, arg in enumerate(args):
            if arg == "--gen" and i + 1 < len(args):
                gen_id = args[i + 1]
        if gen_id is None:
            for i, arg in enumerate(args):
                if arg == "--show" and i + 1 < len(args):
                    next_arg = args[i + 1]
                    if not next_arg.startswith("-"):
                        gen_id = next_arg
        modo_show(gen_id)
        return

    if "--list" in args:
        modo_listar()
        return

    modo_debug()


if __name__ == "__main__":
    main()
