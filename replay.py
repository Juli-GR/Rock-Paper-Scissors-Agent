"""Evalúa agentes en replay sobre partidas reales de humanos (ver rps/human_data.py)."""
import pandas as pd

from rps.agents import MetaAgent, QLearningAgent, RegretMatchingAgent
from rps.env import reward
from rps.human_data import load_games, replay
from rps.opponents import MarkovPlayer

VERSIONS = (1, 2, 3)
# Agentes a evaluar: nombre -> función que crea uno nuevo (uno por partida, cada humano es distinto).
AGENTS = {
    "frequency": lambda: MarkovPlayer(order=0, seed=0),
    "markov1": lambda: MarkovPlayer(order=1, seed=0),
    "markov2": lambda: MarkovPlayer(order=2, seed=0),
    "markov2+own": lambda: MarkovPlayer(order=2, own_order=1, seed=0),
    "qlearning": lambda: QLearningAgent(memory=2, own_memory=1, patience=20, seed=0),
    "regret": lambda: RegretMatchingAgent(seed=0),
    "meta": lambda: MetaAgent(decay=0.99, seed=0),
}


def stats(rewards: list[int]) -> dict:
    """Porcentajes de victorias, empates y derrotas, y score (W - L) cada 100 rondas."""
    n = len(rewards)
    wins, losses = rewards.count(1), rewards.count(-1)
    return {"win%": 100 * wins / n, "tie%": 100 * (n - wins - losses) / n,
            "loss%": 100 * losses / n, "score/100": 100 * (wins - losses) / n}


def evaluate(games) -> pd.DataFrame:
    """Una fila por agente. "2ª mitad" mide solo la segunda mitad de cada partida,
    cuando el agente ya tuvo tiempo de aprender al humano."""
    rows = {}
    # Referencia: lo que consiguió el rival original (el otro humano o el bot).
    named = {"rival original": None, **AGENTS}
    for name, make in named.items():
        all_rewards, second_half = [], []
        for game in games:
            if make is None:
                rewards = [reward(o, h) for o, h in zip(game.opponent, game.human)]
            else:
                rewards = replay(make(), game)
            all_rewards += rewards
            second_half += rewards[len(rewards) // 2:]
        rows[name] = {**stats(all_rewards), "score/100 2ª mitad": stats(second_half)["score/100"]}
    return pd.DataFrame(rows).T.round(1)


def main():
    for version in VERSIONS:
        games = load_games(version)
        rounds = sum(len(g) for g in games)
        print(f"\n=== v{version}: {len(games)} partidas, {rounds} rondas ===")
        print(evaluate(games).to_string())


if __name__ == "__main__":
    main()
