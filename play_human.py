"""Jugá vos contra el agente desde la terminal. Al salir, la partida se registra en MLflow."""
from play import AGENT
from rps.env import play_match
from rps.opponents import Quit, TerminalPlayer
from rps.tracking import log_run


def main():
    agent_class, agent_params = AGENT
    agent = agent_class(**{**agent_params, "seed": None})  # sin semilla: cada partida es distinta
    human = TerminalPlayer()
    print("R rock, P paper, S scissors, Q quit")

    rewards = []  # desde el punto de vista del agente
    try:
        while True:
            rewards += play_match(agent, human, 1)
    except (Quit, KeyboardInterrupt, EOFError):
        pass

    if not rewards:
        return
    log_run({"humano": agent}, {"humano": human}, {"humano": rewards}, rounds=len(rewards),
            run_name="vs humano")


if __name__ == "__main__":
    main()
