"""Oponentes que imitan a jugadores humanos."""
import random
from collections import deque

from ..env import MOVES, Move, Player, beats


class HumanLikePlayer(Player):
    """Mezcla sesgos humanos conocidos, con ruido:

    - leve preferencia por ROCK (`bias`)
    - si gana, tiende a repetir (`stay`); si pierde, a jugar lo que le habría ganado al agente (`shift`)
    - evita repetir la misma jugada 3 veces seguidas (falacia del jugador)
    """

    def __init__(self, bias=(4, 3, 3), stay=0.6, shift=0.6, seed=None):
        self.bias, self.stay, self.shift, self.seed = bias, stay, shift, seed
        self.rng = random.Random(seed)
        self.last = None  # (own, other, reward) de la ronda anterior
        self.own = deque(maxlen=2)

    def act(self) -> Move:
        move = self.rng.choices(MOVES, self.bias)[0]
        if self.last is not None:
            own, other, reward = self.last
            if reward > 0 and self.rng.random() < self.stay:
                move = own
            elif reward < 0 and self.rng.random() < self.shift:
                move = beats(other)
        if len(self.own) == 2 and self.own[0] == self.own[1] == move:
            move = self.rng.choice([m for m in MOVES if m != move])
        return move

    def observe(self, own, other, reward):
        self.last = (own, other, reward)
        self.own.append(own)

    def params(self):
        return {"bias": self.bias, "stay": self.stay, "shift": self.shift, "seed": self.seed}
