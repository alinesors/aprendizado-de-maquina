# Agente Q-Learning Tabular (Aprendizado por Reforco)

[Voltar ao README](../../README.md) &nbsp; | &nbsp; **Paradigma:** Aprendizado por Reforco &nbsp; | &nbsp; **Objetivo:** Aprender politica otima via tentativa e erro

---

## Ideia Central

> O agente aprende por tentativa e erro, atualizando uma tabela Q com 36.720 estados discretizados. A cada acao, recebe recompensa ou penalidade do ambiente. Com 80.000 episodios de treino, a tabela converge para a politica otima.

---

## Loop de Aprendizado

```mermaid
flowchart LR
    A[Agente<br/>Q-Learning] -->|acao a| B[Ambiente<br/>Orbital]
    B -->|obs s', rec r| A
    A --> C[TABELA Q<br/>36.720 estados x 5 acoes]
    C -.->|Q s,a += alpha x r + gamma x max Q s',a' - Q s,a| A
```

---

## Como Executar

| Comando | Descricao |
|---|---|
| `python run_qlearning.py --train` | Treino completo — 80.000 episodios headless |
| `python run_qlearning.py --train --eps N` | Treino com N episodios |
| `python run_qlearning.py --show` | Assiste o agente treinado (1 episodio) |
| `python run_qlearning.py --watch` | Sinonimo de `--show` |
| `python run_qlearning.py --show --episodios 5` | Assiste N episodios |
| `python run_qlearning.py --list` | Lista checkpoints salvos |

> O treino pode levar horas. A tabela Q treinada ja esta incluida em `game-enviroment/agents/q_learning/checkpoints/`.
> A cada 5.000 episodios, o treino exibe um showcase visual para acompanhar o progresso.

---

## Componentes

| Arquivo | Papel |
|---|---|
| `run_qlearning.py` | Entrada unificada: treino, watch e listagem de checkpoints |
| `agents/q_learning/treinador.py` | Loop de treino: 80k episodios + showcases periodicos + reward shaping |
| `agents/q_learning/agent.py` | `AgenteQLearning`: politica epsilon-greedy, Bellman, persistencia |
| `agents/q_learning/discretizer.py` | `DiscretizadorEstado`: 7D continuo -> 6D discreto (36.720 estados) |
| `agents/q_learning/checkpoints/` | Tabelas Q treinadas (dicionario esparso serializado) |

---

## Equacao de Bellman

```
Q(S, A)  <-  Q(S, A)  +  alpha  x  [  R  +  gamma x max Q(S', a)  -  Q(S, A)  ]
   ^            ^         ^        ^        ^     ^            ^
 novo        antigo     taxa     recomp.  desconto  futuro      erro TD
Q-value     Q-value    aprend.  imediata    (y)    estimado
```

| Simbolo | Significado | Valor |
|---|---|---|
| **alpha** | Taxa de aprendizado — confianca em novas experiencias | `0.05` |
| **gamma** | Fator de desconto — valorizacao do futuro | `0.97` |
| **R** | Recompensa imediata do ambiente | variavel |
| **S** | Estado atual (tupla de 6 ints) | — |
| **A** | Acao executada (0 a 4) | — |
| **S'** | Proximo estado | — |

---

## Discretizador (Bussola Dinamica)

> O espaco continuo de **7 dimensoes** e reduzido para **6 inteiros** = **36.720 estados possiveis**.

```mermaid
flowchart LR
    A["7 floats<br/>pos_x, pos_y<br/>vel_x, vel_y<br/>fuel<br/>dist_cp<br/>dist_st"] -->|DiscretizadorEstado| B["6 ints<br/>direcao_alvo (0-15)<br/>direcao_movimento (0-16)<br/>perigo (0-2)<br/>nivel_combustivel (0-4)<br/>distancia (0-2)<br/>magnitude_vel (0-2)"]
```
```
Total: 16x17x3x5x3x3 = 36.720 estados
```

### Detalhamento das Dimensoes

| Dimensao | Faixa | Criterio |
|---|---|---|
| `direcao_alvo` | `0-15` | 16 setores de 22.5° — direcao do alvo atual |
| `direcao_mov` | `0-16` | 16 setores + 1 "parado" (`vel < 0.1`) |
| `perigo` | `0-2` | `0`=seguro, `1`=alerta (`<80px`), `2`=perigo (`<40px`) |
| `nivel_comb` | `0-4` | `0`=critico (`<40`), `1`=baixo (`<80`), `2`=medio (`<120`), `3`=alto (`<160`), `4`=cheio |
| `distancia` | `0-2` | `0`=perto (`<80px`), `1`=medio (`<200px`), `2`=longe |
| `magnitude_vel` | `0-2` | `0`=lento (`<0.8`), `1`=moderado (`<1.8`), `2`=rapido |

### Bussola Dinamica

O alvo e **sempre** o checkpoint nao coletado mais proximo. Quando todos sao coletados, o alvo vira a estacao espacial.

> Isso cria um comportamento natural de varredura: o agente e guiado sequencialmente pelos checkpoints ate o objetivo final.

---

## Politica epsilon-greedy

```mermaid
flowchart TD
    A[Sortear p em 0,1] --> B{p < epsilon?}
    B -->|Sim| C[ALEATORIO<br/>exploracao]
    B -->|Nao| D[GREEDY<br/>argmax Q s,a]
```
```
epsilon decai a cada episodio: epsilon <- max(epsilon x 0.99996, 0.03)
```

---

## Hiperparametros

| Parametro | Valor | Significado |
|---|---|---|
| **alpha** | `0.05` | Taxa de aprendizado |
| **gamma** | `0.97` | Fator de desconto (valoriza futuro) |
| **epsilon** inicial | `1.0` | 100% exploracao no inicio |
| **epsilon** minimo | `0.03` | 3% de exploracao minima |
| **epsilon** decay | `0.99996` | Decaimento exponencial por episodio |
| Valor otimista | `5.0` | Q-values iniciais (incentiva exploracao) |
| Episodios | `80.000` | Total de episodios de treino |
| MAX_PASSOS | `2.000` | Passos maximos por episodio |

---

## Reward Shaping

> Alem das recompensas do ambiente, o agente recebe um **bonus de aproximacao**:

```
bonus = (dist_antiga - dist_nova) x 2.0
```

Se a nave se aproximou do alvo -> bonus positivo. Se se afastou -> bonus negativo. Isso acelera o aprendizado em um ambiente com recompensas naturalmente esparsas.

---

## Inicializacao Otimista

> Todos os Q-values comecam em `5.0` em vez de `0.0`.

O agente assume que **toda acao desconhecida e boa** ate provar o contrario. Isso o incentiva a experimentar estados novos. Acoes que levam a colisoes (`-1000`) tem o Q-value rapidamente reduzido. Acoes que levam a checkpoints (`+500`) mantem o Q-value alto.

---

[Voltar ao README](../../README.md)
