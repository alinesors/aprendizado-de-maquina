import os
import pickle
import time
import numpy as np
import pygame

from .genetic_env import AmbienteGenetico

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

SHOWCASE_DELAY_FRAMES = 120


class TreinadorGenetico:

    def __init__(self, pop_size=200):
        self.pop_size = pop_size

    def treinar(self, geracoes=200, max_steps=2000,
                taxa_mutacao=0.06, forca_mutacao=0.15, decay_forca=0.9992,
                showcase=False):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        for f in os.listdir(CHECKPOINT_DIR):
            if f.endswith(".pkl"):
                os.remove(os.path.join(CHECKPOINT_DIR, f))

        env = AmbienteGenetico(pop_size=self.pop_size, render_mode=None)

        melhor_fitness = 0.0
        melhor_cromossomo = None
        melhor_geracao = 0
        melhor_stats = {}
        ultima_melhoria = 0
        gens_sem_melhoria = 0

        t0 = time.time()

        for gen in range(1, geracoes + 1):
            eh_checkpoint = (gen == 1 or gen % 50 == 0 or gen == geracoes)

            for _ in range(max_steps):
                actions = []
                for n in env.frota:
                    if n.ativa:
                        sensores = env.obter_sensores_nave(n)
                        actions.append(n.cerebro.prever_acao(sensores))
                    else:
                        actions.append(0)
                env.step(actions)
                if env.total_alive == 0:
                    break

            env.calcular_fitness_frota()

            top = max(env.frota, key=lambda n: n.fitness)
            avg = sum(n.fitness for n in env.frota) / len(env.frota)

            if top.fitness > melhor_fitness:
                melhor_fitness = top.fitness
                melhor_cromossomo = top.cerebro.obter_cromossomo().copy()
                melhor_geracao = gen
                ultima_melhoria = gen
                melhor_stats = {
                    "checkpoints": top.checkpoints_coletados,
                    "docked": top.status == "docked",
                    "steps_alive": top.steps_alive,
                    "fuel_restante": float(top.fuel),
                }

            gens_sem_melhoria = gen - ultima_melhoria

            docks = env.total_docked
            cp_best = top.checkpoints_coletados
            alive = env.total_alive

            if gen % 5 == 0 or gen == 1:
                docado = "  *** DOCADO! ***" if docks > 0 else ""
                print("Gen {:3d} | fit={:.0f}  cp={}  dock={}  alive={:2d}  avg={:.0f}{}".format(
                    gen, top.fitness, cp_best, docks, alive, avg, docado))

            if eh_checkpoint:
                crom = top.cerebro.obter_cromossomo()
                self._salvar(gen, crom, top.fitness,
                             top.checkpoints_coletados, top.status == "docked",
                             top.steps_alive, float(top.fuel))

            if eh_checkpoint:
                self._rodar_showcase(env, gen, max_steps)

            pct_renovacao = 0.0
            if gens_sem_melhoria >= 40 and gen > 40:
                pct_renovacao = 0.3
                forca_mutacao = 0.15
                ultima_melhoria = gen
                gens_sem_melhoria = 0
                print(">>> Reinicializacao de diversidade na gen {} (30% novos)".format(gen))

            env.evoluir_geracao(taxa_mutacao, forca_mutacao, pct_renovacao=pct_renovacao)
            env.reset()

            forca_mutacao *= decay_forca

        if melhor_cromossomo is not None:
            self._salvar_melhor(melhor_cromossomo, melhor_fitness, melhor_geracao, melhor_stats)

        t1 = time.time()
        print()
        print("Treino concluido: {} geracoes em {:.0f}s".format(geracoes, t1 - t0))
        print("Melhor: gen {} | fit={:.0f} | cp={} | dock={}".format(
            melhor_geracao, melhor_fitness,
            melhor_stats.get("checkpoints", 0),
            melhor_stats.get("docked", False)))

        env.close()

    def _salvar(self, geracao, cromossomo, fitness, checkpoints, docked, steps_alive, fuel):
        data = {
            "generation": geracao,
            "fitness": fitness,
            "cromossomo": cromossomo,
            "checkpoints": checkpoints,
            "docked": docked,
            "steps_alive": steps_alive,
            "fuel_restante": fuel,
        }
        path = os.path.join(CHECKPOINT_DIR, "gen_{:03d}.pkl".format(geracao))
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _salvar_melhor(self, cromossomo, fitness, geracao, stats):
        data = {
            "generation": geracao,
            "fitness": fitness,
            "cromossomo": cromossomo,
            "checkpoints": stats.get("checkpoints", 0),
            "docked": stats.get("docked", False),
            "steps_alive": stats.get("steps_alive", 0),
            "fuel_restante": stats.get("fuel_restante", 0.0),
        }
        path = os.path.join(CHECKPOINT_DIR, "best.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _rodar_showcase(self, env_headless, geracao, max_steps):
        dnas = [n.cerebro.obter_cromossomo() for n in env_headless.frota]

        env = AmbienteGenetico(pop_size=self.pop_size, render_mode="human")
        env.generation = geracao
        for nave, dna in zip(env.frota, dnas):
            nave.cerebro.definir_cromossomo(dna)

        running = True
        delay = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if env.total_alive > 0 and env.episode_steps < max_steps and delay == 0:
                actions = []
                for n in env.frota:
                    if n.ativa:
                        sensores = env.obter_sensores_nave(n)
                        actions.append(n.cerebro.prever_acao(sensores))
                    else:
                        actions.append(0)
                env.step(actions)
                env.render()
                continue

            if env.total_alive == 0 or env.episode_steps >= max_steps:
                delay += 1
            else:
                delay = 0

            env.render()

            if delay >= SHOWCASE_DELAY_FRAMES:
                running = False

        for hv, vv in zip(env_headless.frota, env.frota):
            hv.ativa = vv.ativa
            hv.status = vv.status
            hv.steps_alive = vv.steps_alive
            hv.checkpoints_coletados = vv.checkpoints_coletados
            hv.fuel = vv.fuel

        env_headless.total_alive = env.total_alive
        env_headless.total_docked = env.total_docked
        env_headless.total_crashed = env.total_crashed
        env_headless.total_oob = env.total_oob
        env_headless.total_no_fuel = env.total_no_fuel
        env_headless.episode_steps = env.episode_steps

        pygame.event.clear()
        if env.screen is not None:
            pygame.display.quit()
            env.screen = None
            env.canvas = None
            env.clock = None

    @staticmethod
    def carregar(gen_id=None):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        if gen_id is not None:
            path = os.path.join(CHECKPOINT_DIR, "gen_{:03d}.pkl".format(int(gen_id)))
        else:
            path = os.path.join(CHECKPOINT_DIR, "best.pkl")

        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def listar():
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoints = []
        for f in sorted(os.listdir(CHECKPOINT_DIR)):
            if f.endswith(".pkl"):
                path = os.path.join(CHECKPOINT_DIR, f)
                try:
                    with open(path, "rb") as fp:
                        data = pickle.load(fp)
                    checkpoints.append({
                        "file": f,
                        "generation": data.get("generation", 0),
                        "fitness": data.get("fitness", 0),
                        "checkpoints": data.get("checkpoints", 0),
                        "docked": data.get("docked", False),
                    })
                except (pickle.UnpicklingError, KeyError, EOFError):
                    pass
        return checkpoints
