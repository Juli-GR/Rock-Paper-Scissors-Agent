"""Oponentes simples: no miran lo que hace el agente."""
import random

from ..env import MOVES, Move, Player

LETTERS = {m.name[0]: m for m in MOVES}  # "R" -> ROCK, "P" -> PAPER, "S" -> SCISSORS


class RandomPlayer(Player):
    """Juega al azar. Con `weights` desiguales es un oponente sesgado."""

    def __init__(self, weights=(1, 1, 1), seed=None):
        self.weights, self.seed = weights, seed
        self.rng = random.Random(seed)

    def act(self) -> Move:
        return self.rng.choices(MOVES, self.weights)[0]

    def params(self):
        return {"weights": self.weights, "seed": self.seed}


class DriftPlayer(Player):
    """Sesgado cuyas probabilidades pasan de `start` a `end` a lo largo de `rounds` rondas."""

    def __init__(self, start=(6, 2, 2), end=(2, 2, 6), rounds=1000, seed=None):
        self.start, self.end, self.rounds, self.seed = start, end, rounds, seed
        self.rng = random.Random(seed)
        self.t = 0

    def act(self) -> Move:
        f = min(self.t / self.rounds, 1)
        self.t += 1
        weights = [a + (b - a) * f for a, b in zip(self.start, self.end)]
        return self.rng.choices(MOVES, weights)[0]

    def params(self):
        return {"start": self.start, "end": self.end, "rounds": self.rounds, "seed": self.seed}


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


class SequencePlayer(Player):
    """Repite una secuencia fija en loop, ej. "RPS" (ciclo) o "RRPSPS"."""

    def __init__(self, sequence="RPS"):
        self.sequence = sequence
        self.moves = [LETTERS[c] for c in sequence]
        self.t = 0

    def act(self) -> Move:
        move = self.moves[self.t % len(self.moves)]
        self.t += 1
        return move

    def params(self):
        return {"sequence": self.sequence}
