"""Oponentes que aprenden del agente."""
import random
from collections import defaultdict, deque

from ..env import MOVES, Move, Player, beats


class MarkovPlayer(Player):
    """Predice la próxima jugada del otro según sus últimas `order` jugadas (y las
    `own_order` propias), y le gana. Sirve como oponente y también como agente.

    Con order=0 solo cuenta frecuencias: le gana a la jugada más común del otro.
    """

    def __init__(self, order=1, own_order=0, seed=None):
        self.order, self.own_order, self.seed = order, own_order, seed
        self.rng = random.Random(seed)
        self.counts = defaultdict(lambda: [0] * len(MOVES))  # contexto -> veces que el otro jugó cada cosa
        self.history = deque(maxlen=order)  # jugadas del otro
        self.own_history = deque(maxlen=own_order)  # jugadas propias

    @property
    def context(self) -> tuple:
        return tuple(self.history), tuple(self.own_history)

    def act(self) -> Move:
        counts = self.counts[self.context]
        best = max(counts)
        predicted = self.rng.choice([m for m in MOVES if counts[m] == best])
        return beats(predicted)

    def observe(self, own, other, reward):
        self.counts[self.context][other] += 1
        self.history.append(other)
        self.own_history.append(own)

    def params(self):
        return {"order": self.order, "own_order": self.own_order, "seed": self.seed}

    def state_dict(self):
        """Los conteos: {"rival ROCK,PAPER | propio SCISSORS": {"ROCK": n, ...}, ...}."""
        def names(moves):
            return ",".join(m.name for m in moves) or "-"

        return {f"rival {names(other)} | propio {names(own)}": dict(zip((m.name for m in MOVES), c))
                for (other, own), c in self.counts.items()}
