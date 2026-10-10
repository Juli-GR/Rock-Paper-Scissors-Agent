"""Enfrenta al agente contra cada oponente, muestra los resultados y los registra en MLflow."""
from rps.agents import QLearningAgent
from rps.env import play_match
from rps.opponents import (DriftPlayer, HumanLikePlayer, MarkovPlayer, Noisy, RandomPlayer,
                           ReactivePlayer, RepeaterPlayer, SequencePlayer, Switcher,
                           WinStayLoseShiftPlayer)
from rps.tracking import LAST, log_run, summarize

ROUNDS = 1000
AGENT = {"memory": 2, "alpha": 0.1, "gamma": 0.0, "epsilon": 0.1, "patience": 20, "seed": 0}


def make_opponents() -> dict:
    """Oponentes nuevos (sin nada aprendido) para un experimento."""
    return {
        # simples
        "random": RandomPlayer(seed=1),
        "biased": RandomPlayer(weights=(6, 2, 2), seed=2),
        "drift": DriftPlayer(rounds=ROUNDS, seed=3),
        "repeater": RepeaterPlayer(seed=4),
        "cycle": SequencePlayer("RPS"),
        "sequence": SequencePlayer("RRPSPS"),
        # reactivos
        "beat_last": ReactivePlayer(shift=1, seed=5),
        "copycat": ReactivePlayer(shift=0, seed=6),
        "delayed": ReactivePlayer(shift=1, delay=3, seed=7),
        "wsls": WinStayLoseShiftPlayer(seed=8),
        # adaptativos
        "frequency": MarkovPlayer(order=0, seed=9),
        "markov": MarkovPlayer(order=1, seed=10),
        "self_play": QLearningAgent(**{**AGENT, "seed": 11}),
        # humano sintético y combinados
        "human": HumanLikePlayer(seed=12),
        "noisy_cycle": Noisy(SequencePlayer("RPS"), p=0.2, seed=13),
        "switcher": Switcher([SequencePlayer("RPS"), RandomPlayer(weights=(6, 2, 2), seed=14),
                              ReactivePlayer(shift=1, seed=15)], every=ROUNDS // 4),
    }


def main():
    opponents = make_opponents()
    agents, results = {}, {}
    for name, opponent in opponents.items():
        agent = agents[name] = QLearningAgent(**AGENT)  # uno nuevo por oponente
        results[name] = play_match(agent, opponent, ROUNDS)
        s = summarize(results[name])
        print(f"{name:>11}: W {s['wins']:4}  T {s['ties']:4}  L {s['losses']:4}  "
              f"score {s['score']:+5d}  últimas {LAST}: {s[f'score_last_{LAST}']:+4d}")
    log_run(agents, opponents, results, rounds=ROUNDS)


if __name__ == "__main__":
    main()
