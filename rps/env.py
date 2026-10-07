"""Reglas del juego, la interfaz común de jugadores y el loop de partidas."""
from abc import ABC, abstractmethod
from enum import IntEnum


class Move(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


MOVES = list(Move)


def reward(a: Move, b: Move) -> int:
    """+1 si `a` le gana a `b`, 0 si empatan, -1 si pierde."""
    # Cada jugada le gana a la anterior (mod 3): PAPER > ROCK, SCISSORS > PAPER, ROCK > SCISSORS.
    return (0, 1, -1)[(a - b) % 3]


class Player(ABC):
    """Todo lo que juega (oponentes programados y, más adelante, agentes de RL)."""

    @abstractmethod
    def act(self) -> Move:
        """Elige la jugada de esta ronda."""

    def observe(self, own: Move, other: Move, reward: int) -> None:
        """Recibe el resultado de la ronda. Por defecto no hace nada."""

    def params(self) -> dict:
        """Configuración que define al jugador (para registrar experimentos)."""
        return {}

    def state_dict(self) -> dict:
        """Lo aprendido, serializable a JSON. Vacío si no aprende nada."""
        return {}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"


def play_match(p1: Player, p2: Player, rounds: int) -> list[int]:
    """Juega `rounds` rondas y devuelve las rewards desde el punto de vista de `p1`."""
    rewards = []
    for _ in range(rounds):
        m1, m2 = p1.act(), p2.act()
        r = reward(m1, m2)
        p1.observe(m1, m2, r)
        p2.observe(m2, m1, -r)
        rewards.append(r)
    return rewards
