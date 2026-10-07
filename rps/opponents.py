"""Oponentes programados para entrenar y evaluar agentes."""
import random

from .env import MOVES, Move, Player


class RandomPlayer(Player):
    """Juega al azar. Con `weights` desiguales es un oponente sesgado."""

    def __init__(self, weights=(1, 1, 1), seed=None):
        self.weights, self.seed = weights, seed
        self.rng = random.Random(seed)

    def act(self) -> Move:
        return self.rng.choices(MOVES, self.weights)[0]

    def params(self):
        return {"weights": self.weights, "seed": self.seed}


class RepeaterPlayer(Player):
    """Repite siempre su jugada anterior (la primera es aleatoria)."""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.last = None

    def act(self) -> Move:
        return self.last if self.last is not None else self.rng.choice(MOVES)

    def observe(self, own, other, reward):
        self.last = own

    def params(self):
        return {"seed": self.seed}


class CyclePlayer(Player):
    """Recorre ROCK → PAPER → SCISSORS → ROCK..."""

    def __init__(self, start=Move.ROCK):
        self.start = self.next = start

    def act(self) -> Move:
        move, self.next = self.next, Move((self.next + 1) % 3)
        return move

    def params(self):
        return {"start": self.start.name}
