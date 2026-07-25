"""
TreinadorQLearning — Loop de treino para o agente Q-Learning Tabular.

Recompensas originais do ambiente + reward shaping por aproximacao.
Discretizador de 36720 estados (6 dimensoes).

Treino longo: 80000 episodios headless.
Showcase: 1 episodio visual greedy a cada 5000 ep + 5 no final.
"""

import os
import sys
import time

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from orbital_env import OrbitalEnv

from .discretizer import DiscretizadorEstado
from .agent import AgenteQLearning

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

ICONES = {
    "docked": "ATRACOU", "crashed_planet": "COLIDIU",
    "out_of_bounds": "FORA", "no_fuel": "SEM COMB.",
    "timeout": "TIMEOUT",
}


def treinar(n_episodios=80000, alpha=0.05, gamma=0.97, epsilon=1.0,
            epsilon_min=0.03, epsilon_decay=0.99996, valor_otimista=5.0,
            log_intervalo=500, showcase_intervalo=5000, max_passos=2000,
            n_showcase=5):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 60)
    print("  ODISSEIA ORBITAL — Q-Learning Tabular")
    print("=" * 60)
    print(f"  Estados        : {DiscretizadorEstado.TOTAL_ESTADOS}")
    print(f"  Episodios      : {n_episodios}")
    print(f"  Recompensa     : ambiente + reward shaping (aproximacao)")
    print(f"  alpha={alpha}  gamma={gamma}  eps_min={epsilon_min}  eps_decay={epsilon_decay}")
    print(f"  Eq. Bellman    : Q(S,A) += a * [R + g * max Q(S',a) - Q(S,A)]")
    print("=" * 60)

    env = OrbitalEnv(render_mode=None)
    disc = DiscretizadorEstado(
        checkpoints=env.checkpoints,
        posicao_estacao=env.station_pos,
        planetas=env.planets,
    )
    agente = AgenteQLearning(
        n_acoes=5,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        valor_otimista=valor_otimista,
    )

    print(f"\n[Treinando {n_episodios} episodios...]")
    print("-" * 60)

    t0 = time.time()
    sucessos_janela = 0
    recs_janela = []
    max_cps = 0

    for ep in range(n_episodios):
        obs = env.reset()
        estado = disc.discretizar(obs)
        terminal = False
        ep_rec = 0.0
        ep_pas = 0
        info = {}
        ep_cps = 0

        while not terminal and ep_pas < max_passos:
            # 1. ANTES DA ACAO: calcula distancia ate o alvo atual
            pos_nave_atual = np.array([obs[0], obs[1]], dtype=np.float64)
            alvo_atual = disc._encontrar_alvo(pos_nave_atual)
            dist_antiga = float(np.linalg.norm(pos_nave_atual - alvo_atual))

            # 2. SELECIONA E EXECUTA A ACAO
            acao = agente.selecionar_acao(estado)
            prox_obs, recompensa_ambiente, terminal, info = env.step(acao)

            # 3. DEPOIS DA ACAO: calcula nova distancia em relacao ao MESMO alvo
            pos_nave_nova = np.array([prox_obs[0], prox_obs[1]], dtype=np.float64)
            dist_nova = float(np.linalg.norm(pos_nave_nova - alvo_atual))

            # 4. REWARD SHAPING: bonus por se aproximar do alvo
            delta_distancia = dist_antiga - dist_nova
            bonus_aproximacao = delta_distancia * 2.0
            recompensa_total = recompensa_ambiente + bonus_aproximacao

            # 5. ATUALIZACAO E APRENDIZADO
            prox_estado = disc.discretizar(prox_obs)
            agente.aprender(estado, acao, recompensa_total, prox_estado, terminal)

            if info.get("checkpoint_collected"):
                ep_cps += 1

            estado = prox_estado
            obs = prox_obs
            ep_rec += recompensa_total
            ep_pas += 1

        if ep_pas >= max_passos:
            info = {"status": "timeout"}

        if ep_cps > max_cps:
            max_cps = ep_cps
        recs_janela.append(ep_rec)
        if info.get("status") == "docked":
            sucessos_janela += 1
        agente.decair_epsilon()

        if (ep + 1) % log_intervalo == 0:
            tx = sucessos_janela / log_intervalo * 100
            max_rec = max(recs_janela)
            decorrido = time.time() - t0
            print(
                f"  Ep {ep + 1:5d} | Sucesso: {tx:5.1f}% | "
                f"maxRec: {max_rec:9.1f} | Eps: {agente.epsilon:.4f} | "
                f"maxCPs: {max_cps:1d} | Q:{len(agente.tabela_q):4d} | {decorrido:.0f}s"
            )
            sucessos_janela = 0
            recs_janela = []

        if ep == 0 or (ep + 1) % showcase_intervalo == 0:
            epsilon_salvo = agente.epsilon
            agente.epsilon = 0.0
            env_show = OrbitalEnv(render_mode="human")
            disc_show = DiscretizadorEstado(
                checkpoints=env_show.checkpoints,
                posicao_estacao=env_show.station_pos,
                planetas=env_show.planets,
            )
            obs = env_show.reset()
            estado = disc_show.discretizar(obs)
            terminal = False
            ep_rec = 0.0
            ep_pas = 0
            info = {}
            ep_cps = 0
            while not terminal and ep_pas < max_passos:
                acao = agente.selecionar_acao(estado)
                prox_obs, recompensa, terminal, info = env_show.step(acao)
                prox_estado = disc_show.discretizar(prox_obs)
                if info.get("checkpoint_collected"):
                    ep_cps += 1
                estado = prox_estado
                ep_rec += recompensa
                ep_pas += 1
                env_show.render()
            icone = ICONES.get(info.get("status", "timeout"), "???")
            print(f"\n  >>> Showcase Ep {ep + 1} (greedy) <<<")
            print(f"      {icone:10s} | Rec: {ep_rec:8.1f} | Passos: {ep_pas:4d} | CPs: {ep_cps}\n")
            env_show.close()
            agente.epsilon = epsilon_salvo

    tempo_total = time.time() - t0
    env.close()

    print("-" * 60)
    print(f"[Treino concluido em {tempo_total:.0f}s]")
    print(f"  Epsilon final: {agente.epsilon:.4f}")
    print(f"  Estados Q: {len(agente.tabela_q)} de {DiscretizadorEstado.TOTAL_ESTADOS}")
    print(f"  Recorde de CPs: {max_cps}")

    # Showcase
    print(f"\n[SHOWCASE] epsilon=0, visual...")
    print("-" * 60)

    agente.epsilon = 0.0
    env_show = OrbitalEnv(render_mode="human")

    for ep in range(n_showcase):
        obs = env_show.reset()
        estado = disc.discretizar(obs)
        terminal = False
        ep_rec = 0.0
        ep_pas = 0
        info = {}
        ep_cps = 0

        while not terminal and ep_pas < max_passos:
            acao = agente.selecionar_acao(estado)
            prox_obs, recompensa, terminal, info = env_show.step(acao)
            prox_estado = disc.discretizar(prox_obs)
            if info.get("checkpoint_collected"):
                ep_cps += 1
            estado = prox_estado
            ep_rec += recompensa
            ep_pas += 1
            env_show.render()

        icone = ICONES.get(info.get("status", "timeout"), "???")
        print(f"  {ep + 1}/{n_showcase} {icone:10s} | Rec: {ep_rec:8.1f} | Passos: {ep_pas:4d} | CPs: {ep_cps}")

    env_show.close()

    ckpt = os.path.join(CHECKPOINT_DIR, 'q_table_odisseia.pkl')
    agente.salvar(ckpt)
    print("-" * 60)
    print(f"[SALVO] {ckpt}")
    print("=" * 60)

    return agente, disc

def assistir(episodios=1, max_passos=2000):
    env = OrbitalEnv(render_mode="human")
    disc = DiscretizadorEstado(
        checkpoints=env.checkpoints,
        posicao_estacao=env.station_pos,
        planetas=env.planets,
    )

    agente = AgenteQLearning(n_acoes=5)
    ckpt = os.path.join(CHECKPOINT_DIR, 'q_table_odisseia.pkl')
    agente.carregar(ckpt)
    agente.epsilon = 0.0

    print("=" * 50)
    print("  ODISSEIA ORBITAL — Assistindo Agente Treinado")
    print("=" * 50)
    print(f"  Q-table carregada: {len(agente.tabela_q)} estados")
    print("-" * 50)

    obs = env.reset()
    estado = disc.discretizar(obs)
    ep_rec = 0.0
    ep_pas = 0
    ep_cps = 0
    info_printed = False
    ep_count = 0
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
                    estado = disc.discretizar(obs)
                    ep_rec = 0.0
                    ep_pas = 0
                    ep_cps = 0
                    info_printed = False

        if not env.done and ep_pas < max_passos:
            acao = agente.selecionar_acao(estado)
            prox_obs, recompensa, terminal, info = env.step(acao)
            prox_estado = disc.discretizar(prox_obs)
            if info.get("checkpoint_collected"):
                ep_cps += 1
            estado = prox_estado
            obs = prox_obs
            ep_rec += recompensa
            ep_pas += 1

        env.render()

        if env.done and not info_printed:
            icone = ICONES.get(env.last_info.get("status", "timeout"), "???")
            ep_count += 1
            collected = sum(1 for cp in env.checkpoints if cp["collected"])
            print(f"  Ep {ep_count:2d} | {icone:10s} | Rec: {ep_rec:8.1f} | Passos: {ep_pas:4d} | CPs: {collected}")
            info_printed = True
            if ep_count >= episodios:
                running = False

    env.close()
    print("-" * 50)
    print("  Fim da exibicao.")
    print("=" * 50)

def listar():
    import glob
    arquivos = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pkl"))
    if arquivos:
        print("Checkpoints salvos:")
        for a in sorted(arquivos):
            nome = os.path.basename(a)
            tamanho = os.path.getsize(a)
            print(f"  {nome:30s}  {tamanho:>8,} bytes")
    else:
        print("Nenhum checkpoint salvo.")

if __name__ == '__main__':
    treinar()
