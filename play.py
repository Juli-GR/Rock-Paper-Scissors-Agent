"""Enfrenta a un jugador contra cada oponente y muestra los resultados."""
from collections import Counter

from rps.agents import QLearningAgent
from rps.env import play_match
from rps.opponents import CyclePlayer, RandomPlayer, RepeaterPlayer
from rps.plots import plot_results

ROUNDS = 1000
LAST = 200  # ventana final para ver qué tan bien juega una vez que aprendió
PLOT_PATH = "results.png"


def main():
    opponents = {
        "random": RandomPlayer(seed=1),
        "biased": RandomPlayer(weights=(6, 2, 2), seed=2),
        "repeater": RepeaterPlayer(seed=3),
        "cycle": CyclePlayer(),
    }
    results = {}
    for name, opponent in opponents.items():
        agent = QLearningAgent(patience=20, seed=0)
        rewards = results[name] = play_match(agent, opponent, ROUNDS)
        c = Counter(rewards)
        last = sum(rewards[-LAST:])
        print(f"{name:>8}: W {c[1]:4}  T {c[0]:4}  L {c[-1]:4}  "
              f"score {c[1] - c[-1]:+5d}  últimas {LAST}: {last:+d}")
    plot_results(results, path=PLOT_PATH)
    print(f"Gráfico guardado en {PLOT_PATH}")


if __name__ == "__main__":
    main()
