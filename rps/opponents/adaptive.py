"""Oponentes que aprenden del agente."""
import random
from collections import defaultdict, deque

from ..env import MOVES, Move, Player, beats


class MarkovPlayer(Player):
    """Predice la próxima jugada del agente según sus últimas `order` jugadas, y le gana.

    Con order=0 solo cuenta frecuencias: le gana a la jugada más común del agente.
    """

    def __init__(self, order=1, seed=None):
        self.order, self.seed = order, seed
        self.rng = random.Random(seed)
        self.counts = defaultdict(lambda: [0] * len(MOVES))  # contexto -> veces que el agente jugó cada cosa
        self.history = deque(maxlen=order)

    def act(self) -> Move:
        counts = self.counts[tuple(self.history)]
        best = max(counts)
        predicted = self.rng.choice([m for m in MOVES if counts[m] == best])
        return beats(predicted)

    def observe(self, own, other, reward):
        self.counts[tuple(self.history)][other] += 1
        self.history.append(other)

    def params(self):
        return {"order": self.order, "seed": self.seed}
