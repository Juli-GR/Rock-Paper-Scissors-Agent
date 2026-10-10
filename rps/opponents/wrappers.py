"""Combinadores: arman oponentes nuevos a partir de otros."""
import random

from ..env import MOVES, Move, Player


class Noisy(Player):
    """Juega como `opponent`, pero con probabilidad `p` juega al azar."""

    def __init__(self, opponent: Player, p=0.2, seed=None):
        self.opponent, self.p, self.seed = opponent, p, seed
        self.rng = random.Random(seed)

    def act(self) -> Move:
        move = self.opponent.act()  # siempre se llama, para que el interno no pierda el ritmo
        return self.rng.choice(MOVES) if self.rng.random() < self.p else move

    def observe(self, own, other, reward):
        self.opponent.observe(own, other, reward)

    def params(self):
        return {"opponent": self.opponent, "p": self.p, "seed": self.seed}


class Switcher(Player):
    """Cambia de estrategia cada `every` rondas, rotando entre `opponents`.

    Todos observan cada ronda, para que el que entra esté al día con la partida.
    """

    def __init__(self, opponents: list[Player], every=250):
        self.opponents, self.every = opponents, every
        self.t = 0

    def act(self) -> Move:
        current = self.opponents[self.t // self.every % len(self.opponents)]
        self.t += 1
        return current.act()

    def observe(self, own, other, reward):
        for o in self.opponents:
            o.observe(own, other, reward)

    def params(self):
        return {"opponents": self.opponents, "every": self.every}
