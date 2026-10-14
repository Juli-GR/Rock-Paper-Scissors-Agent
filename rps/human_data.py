"""Partidas reales de humanos (github.com/erik-brockbank/rps) y replay de agentes sobre ellas.

v1: humano vs humano. v2: humano vs bot con estrategia fija. v3: humano vs bot que aprende.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .env import Move, Player, reward

# El repo clonado está al lado de este: Code/RL/rps/analysis.
DATA_DIR = Path(__file__).resolve().parents[2] / "rps" / "analysis"
MOVE_NAMES = {"rock": Move.ROCK, "paper": Move.PAPER, "scissors": Move.SCISSORS}


@dataclass
class Game:
    """Una partida desde el punto de vista del humano: `human[t]` y `opponent[t]` son
    las jugadas de la ronda t."""
    version: int
    game_id: str
    human_id: str
    opponent_type: str  # "human" en v1, la estrategia del bot en v2 y v3
    human: list[Move]
    opponent: list[Move]

    def __len__(self):
        return len(self.human)


def load_games(version: int, min_rounds: int = 100, data_dir: Path = DATA_DIR) -> list[Game]:
    """Carga las partidas de `rps_v{version}_data.csv`.

    Las rondas en que alguno no jugó a tiempo ("none") se saltan, y se descartan las
    partidas que quedan con menos de `min_rounds` rondas.
    En v1 cada partida sale dos veces, una desde el punto de vista de cada humano.
    """
    df = pd.read_csv(data_dir / f"rps_v{version}_data.csv")
    is_bot = df["is_bot"] if "is_bot" in df else pd.Series(0, index=df.index)
    games = []
    for game_id, g in df.groupby("game_id"):
        players = g.groupby("player_id")
        if players.ngroups != 2:
            continue
        # Una columna de jugadas por jugador, alineadas por ronda.
        moves = g.pivot_table(index="round_index", columns="player_id",
                              values="player_move", aggfunc="first").dropna()
        moves = moves[moves.isin(list(MOVE_NAMES)).all(axis=1)]
        if len(moves) < min_rounds:
            continue
        bots = set(g.loc[is_bot[g.index] == 1, "player_id"])
        humans = [p for p in moves.columns if p not in bots]
        opponent_type = "human" if version == 1 else g["bot_strategy"].iloc[0]
        for human_id in humans:
            other_id = next(p for p in moves.columns if p != human_id)
            games.append(Game(version, game_id, human_id, opponent_type,
                              [MOVE_NAMES[m] for m in moves[human_id]],
                              [MOVE_NAMES[m] for m in moves[other_id]]))
    return games


def replay(agent: Player, game: Game) -> list[int]:
    """Pone a `agent` en el lugar del rival original de `game` y devuelve sus rewards.

    En cada ronda el agente elige su jugada y se compara con lo que jugó el humano. Esa
    reward es válida: los dos eligen a la vez, así que el humano habría jugado lo mismo.
    Pero lo que el agente observa es la ronda grabada (la jugada del rival original),
    no la suya: es la historia que el humano tuvo delante. El futuro con la jugada del
    agente no existe en los datos.
    """
    rewards = []
    for human, recorded in zip(game.human, game.opponent):
        rewards.append(reward(agent.act(), human))
        agent.observe(recorded, human, reward(recorded, human))
    return rewards
