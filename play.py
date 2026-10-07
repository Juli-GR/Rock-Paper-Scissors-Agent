"""Enfrenta al agente contra cada oponente, muestra los resultados y los registra en MLflow."""
from rps.agents import QLearningAgent
from rps.env import play_match
from rps.opponents import CyclePlayer, RandomPlayer, RepeaterPlayer
from rps.tracking import LAST, log_run, summarize

ROUNDS = 1000
AGENT = {"memory": 2, "alpha": 0.1, "gamma": 0.0, "epsilon": 0.1, "patience": 20, "seed": 0}


def main():
    opponents = {
        "random": RandomPlayer(seed=1),
        "biased": RandomPlayer(weights=(6, 2, 2), seed=2),
        "repeater": RepeaterPlayer(seed=3),
        "cycle": CyclePlayer(),
    }
    agents, results = {}, {}
    for name, opponent in opponents.items():
        agent = agents[name] = QLearningAgent(**AGENT)  # uno nuevo por oponente
        results[name] = play_match(agent, opponent, ROUNDS)
        s = summarize(results[name])
        print(f"{name:>8}: W {s['wins']:4}  T {s['ties']:4}  L {s['losses']:4}  "
              f"score {s['score']:+5d}  últimas {LAST}: {s[f'score_last_{LAST}']:+d}")
    log_run(agents, opponents, results, rounds=ROUNDS)


if __name__ == "__main__":
    main()
