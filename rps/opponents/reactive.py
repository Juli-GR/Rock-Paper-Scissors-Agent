"""Oponentes que reaccionan a las jugadas del agente."""
import random
from collections import deque

from ..env import MOVES, Move, Player, beats


class ReactivePlayer(Player):
    """Responde a la jugada del agente de hace `delay` rondas, corrida `shift` lugares.

    shift=0 la copia, shift=1 juega lo que le gana, shift=2 juega lo que le pierde.
    """

    def __init__(self, shift=1, delay=1, seed=None):
        self.shift, self.delay, self.seed = shift, delay, seed
        self.rng = random.Random(seed)
        self.seen = deque(maxlen=delay)  # seen[0] es la jugada de hace `delay` rondas

    def act(self) -> Move:
        if len(self.seen) < self.delay:
            return self.rng.choice(MOVES)
        return Move((self.seen[0] + self.shift) % 3)

    def observe(self, own, other, reward):
        self.seen.append(other)

    def params(self):
        return {"shift": self.shift, "delay": self.delay, "seed": self.seed}


class WinStayLoseShiftPlayer(Player):
    """Si ganó, repite. Si no, juega lo que le gana a la última jugada del agente."""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.last = None  # (own, other, reward) de la ronda anterior

    def act(self) -> Move:
        if self.last is None:
            return self.rng.choice(MOVES)
        own, other, reward = self.last
        return own if reward > 0 else beats(other)

    def observe(self, own, other, reward):
        self.last = (own, other, reward)

    def params(self):
        return {"seed": self.seed}
