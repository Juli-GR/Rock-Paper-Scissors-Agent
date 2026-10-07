"""Agentes que aprenden jugando."""
import random
from collections import defaultdict, deque

from .env import MOVES, Move, Player


class QLearningAgent(Player):
    """Q-learning tabular. El estado son las últimas `memory` jugadas del oponente.

    Con `patience`, deja de explorar tras `patience` victorias seguidas y vuelve a
    explorar en cuanto no gana una ronda.
    """

    def __init__(self, memory=2, alpha=0.1, gamma=0.0, epsilon=0.1, patience=None, seed=None):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.patience = patience
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: [0.0] * len(MOVES))  # estado -> valor de cada acción
        self.history = deque(maxlen=memory)
        self.streak = 0  # victorias seguidas

    @property
    def state(self) -> tuple:
        return tuple(self.history)

    @property
    def exploring(self) -> bool:
        return self.patience is None or self.streak < self.patience

    def act(self) -> Move:
        if self.exploring and self.rng.random() < self.epsilon:  # explorar
            return self.rng.choice(MOVES)
        values = self.q[self.state]  # explotar (desempate al azar)
        best = max(values)
        return self.rng.choice([m for m in MOVES if values[m] == best])

    def observe(self, own, other, reward):
        state = self.state
        self.history.append(other)
        target = reward + self.gamma * max(self.q[self.state])
        self.q[state][own] += self.alpha * (target - self.q[state][own])
        self.streak = self.streak + 1 if reward > 0 else 0
