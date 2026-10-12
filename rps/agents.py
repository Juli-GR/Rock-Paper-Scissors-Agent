"""Agentes que aprenden jugando."""
import random
from collections import defaultdict, deque

from .env import MOVES, Move, Player, reward
from .opponents import MarkovPlayer


class QLearningAgent(Player):
    """Q-learning tabular. El estado son las últimas `memory` jugadas del oponente y
    las últimas `own_memory` jugadas propias.

    own_memory: con > 0 puede ver a los oponentes que reaccionan a lo que él juega, y
    como su jugada cambia el estado siguiente, gamma > 0 pasa a tener sentido.
    alpha: velocidad con la que aprende (cuánto pesa lo nuevo frente a lo que ya sabía).
    gamma: horizonte de lo que le importa (solo esta ronda, o también las siguientes).

    patience: deja de explorar tras `patience` victorias seguidas y vuelve a
    explorar en cuanto no gana una ronda.
    """

    def __init__(self, memory=2, own_memory=0, alpha=0.1, gamma=0.0, epsilon=0.1,
                 patience=None, seed=None):
        self.memory, self.own_memory = memory, own_memory
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.patience, self.seed = patience, seed
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: [0.0] * len(MOVES))  # estado -> valor de cada acción
        self.history = deque(maxlen=memory)  # jugadas del oponente
        self.own_history = deque(maxlen=own_memory)  # jugadas propias
        self.streak = 0  # victorias seguidas

    @property
    def state(self) -> tuple:
        return tuple(self.history), tuple(self.own_history)

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
        self.own_history.append(own)
        target = reward + self.gamma * max(self.q[self.state])
        self.q[state][own] += self.alpha * (target - self.q[state][own])
        self.streak = self.streak + 1 if reward > 0 else 0

    def params(self):
        return {"memory": self.memory, "own_memory": self.own_memory, "alpha": self.alpha,
                "gamma": self.gamma, "epsilon": self.epsilon, "patience": self.patience,
                "seed": self.seed}

    def state_dict(self):
        """La Q-table: {"rival ROCK,PAPER | propio SCISSORS": {"ROCK": q, ...}, ...}."""
        def names(moves):
            return ",".join(m.name for m in moves) or "-"

        def key(state):
            other, own = state
            return f"rival {names(other)} | propio {names(own)}" if self.own_memory else names(other)

        return {key(state): {m.name: round(v, 4) for m, v in zip(MOVES, values)}
                for state, values in self.q.items()}


class RegretMatchingAgent(Player):
    """Regret matching. Política estocástica: en cada estado juega cada acción con una
    probabilidad, en lugar de elegir siempre la mejor.

    El "regret" de una acción es cuánto más habría ganado jugándola en lugar de lo que
    jugó. Se acumula por estado, y juega cada acción con probabilidad proporcional a su
    regret positivo (si ninguno es positivo, 1/3 cada una).

    El estado es el mismo que en QLearningAgent: las últimas `memory` jugadas del
    oponente y las últimas `own_memory` propias. No necesita explorar: siempre ve qué
    jugó el oponente, así que puede calcular el regret de las 3 acciones.

    decay: olvido. Antes de sumar el regret nuevo, multiplica los del estado por `decay`,
    así lo viejo pesa cada vez menos y se adapta más rápido si el oponente cambia.
    Con 1.0 no olvida nada.
    """

    def __init__(self, memory=2, own_memory=1, decay=1.0, seed=None):
        self.memory, self.own_memory, self.decay, self.seed = memory, own_memory, decay, seed
        self.rng = random.Random(seed)
        self.regrets = defaultdict(lambda: [0.0] * len(MOVES))  # estado -> regret de cada acción
        self.history = deque(maxlen=memory)  # jugadas del oponente
        self.own_history = deque(maxlen=own_memory)  # jugadas propias

    @property
    def state(self) -> tuple:
        return tuple(self.history), tuple(self.own_history)

    def policy(self, state) -> list[float]:
        """Probabilidad de jugar cada acción en `state`."""
        positive = [max(r, 0.0) for r in self.regrets[state]]
        total = sum(positive)
        return [p / total for p in positive] if total > 0 else [1 / len(MOVES)] * len(MOVES)

    def act(self) -> Move:
        return self.rng.choices(MOVES, weights=self.policy(self.state))[0]

    def observe(self, own, other, reward_):
        regrets = self.regrets[self.state]
        for m in MOVES:
            regrets[m] = self.decay * regrets[m] + reward(m, other) - reward_
        self.history.append(other)
        self.own_history.append(own)

    def params(self):
        return {"memory": self.memory, "own_memory": self.own_memory, "decay": self.decay,
                "seed": self.seed}

    def state_dict(self):
        """La política: {"rival ROCK,PAPER | propio SCISSORS": {"ROCK": p, ...}, ...}."""
        def names(moves):
            return ",".join(m.name for m in moves) or "-"

        return {f"rival {names(other)} | propio {names(own)}":
                {m.name: round(p, 4) for m, p in zip(MOVES, self.policy((other, own)))}
                for other, own in self.regrets}


class MetaAgent(Player):
    """Meta-agente estilo Iocaine Powder: tiene varias estrategias adentro y en cada ronda
    juega la que mejor viene funcionando.

    Cada estrategia propone una jugada `m`, y de cada una salen 3 candidatas:
      +0  jugar `m` tal cual
      +1  jugar lo que le gana a `m` (por si el rival ya predijo `m` y va a jugar lo que le gana)
      +2  jugar lo que le gana a eso (un nivel más de "él sabe que yo sé")

    Como siempre ve qué jugó el rival, puede calcular cuánto habría ganado cada candidata,
    aunque no la haya jugado. Lleva ese puntaje con olvido (`decay`) y juega la mejor.

    Todas las estrategias observan lo que realmente pasó (la jugada del meta-agente), así
    que siguen aprendiendo aunque no las elija.
    """

    def __init__(self, decay=0.9, seed=None):
        self.decay, self.seed = decay, seed
        self.rng = random.Random(seed)
        sub = lambda: self.rng.randrange(2**32)  # una semilla distinta para cada estrategia
        self.strategies = {
            "frequency": MarkovPlayer(order=0, seed=sub()),
            "markov1": MarkovPlayer(order=1, seed=sub()),
            "markov2+own": MarkovPlayer(order=2, own_order=1, seed=sub()),
            "regret": RegretMatchingAgent(seed=sub()),
            "regret_decay": RegretMatchingAgent(decay=0.95, seed=sub()),
            "qlearning": QLearningAgent(memory=2, own_memory=1, patience=20, seed=sub()),
        }
        self.candidates = [(name, shift) for name in self.strategies for shift in range(3)]
        self.scores = {c: 0.0 for c in self.candidates}
        self.used = {c: 0 for c in self.candidates}  # veces que se jugó cada candidata
        self.proposals = {}

    def act(self) -> Move:
        moves = {name: s.act() for name, s in self.strategies.items()}
        self.proposals = {(name, shift): Move((moves[name] + shift) % 3)
                          for name, shift in self.candidates}
        best = max(self.scores.values())
        choice = self.rng.choice([c for c in self.candidates if self.scores[c] == best])
        self.used[choice] += 1
        return self.proposals[choice]

    def observe(self, own, other, reward_):
        for c, move in self.proposals.items():
            self.scores[c] = self.decay * self.scores[c] + reward(move, other)
        for s in self.strategies.values():
            s.observe(own, other, reward_)

    def params(self):
        return {"decay": self.decay, "strategies": ",".join(self.strategies), "seed": self.seed}

    def state_dict(self):
        """Cuántas rondas jugó cada candidata, de la más usada a la menos usada."""
        used = sorted(self.used.items(), key=lambda kv: -kv[1])
        return {f"{name} +{shift}": n for (name, shift), n in used if n}
