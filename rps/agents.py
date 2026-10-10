"""Agentes que aprenden jugando."""
import random
from collections import defaultdict, deque

from .env import MOVES, Move, Player


class QLearningAgent(Player):
    """Q-learning tabular. El estado son las últimas `memory` jugadas del oponente.

    alpha: velocidad con la que aprende (cuánto pesa lo nuevo frente a lo que ya sabía).
    gamma: horizonte de lo que le importa (solo esta ronda, o también las siguientes).

    patience: deja de explorar tras `patience` victorias seguidas y vuelve a
    explorar en cuanto no gana una ronda.
    """

    def __init__(self, memory=2, alpha=0.1, gamma=0.0, epsilon=0.1, patience=None, seed=None):
        self.memory, self.alpha, self.gamma, self.epsilon = memory, alpha, gamma, epsilon
        self.patience, self.seed = patience, seed
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

    def params(self):
        return {"memory": self.memory, "alpha": self.alpha, "gamma": self.gamma,
                "epsilon": self.epsilon, "patience": self.patience, "seed": self.seed}

    def state_dict(self):
        """La Q-table: {"ROCK,PAPER": {"ROCK": q, "PAPER": q, "SCISSORS": q}, ...}."""
        return {",".join(m.name for m in state) or "(sin historia)":
                {m.name: round(v, 4) for m, v in zip(MOVES, values)}
                for state, values in self.q.items()}
