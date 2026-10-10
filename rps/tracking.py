"""Registro de experimentos en MLflow."""
import matplotlib.pyplot as plt
import mlflow

from .env import Player
from .plots import blocks, plot_results

EXPERIMENT = "rps"
BLOCK = 20  # rondas por bloque en las curvas y el gráfico
LAST = 200  # ventana final para medir qué tan bien juega una vez que aprendió


def summarize(rewards: list[int]) -> dict:
    wins, ties, losses = rewards.count(1), rewards.count(0), rewards.count(-1)
    return {"wins": wins, "ties": ties, "losses": losses,
            "score": wins - losses, f"score_last_{LAST}": sum(rewards[-LAST:])}


def log_run(agents: dict[str, Player], opponents: dict[str, Player],
            results: dict[str, list[int]], rounds: int, run_name: str | None = None):
    """Registra un run: el agente usado, contra quién jugó, métricas, curvas, gráfico y lo aprendido.

    `agents`, `opponents` y `results` están indexados por el nombre del oponente.
    """
    mlflow.set_experiment(EXPERIMENT)
    agent = next(iter(agents.values()))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"agent": type(agent).__name__, **agent.params(),
                           "rounds": rounds, "opponents": ",".join(opponents)})
        mlflow.log_params({f"opponent.{name}": repr(o) for name, o in opponents.items()})

        summaries = {name: summarize(rewards) for name, rewards in results.items()}
        mlflow.log_metrics({f"{name}.{k}": v for name, s in summaries.items() for k, v in s.items()})
        # Resumen de todo el run, para comparar runs de un vistazo.
        mlflow.log_metric(f"mean_score_last_{LAST}",
                          sum(s[f"score_last_{LAST}"] for s in summaries.values()) / len(summaries))

        curves = {name: blocks(rewards, BLOCK) for name, rewards in results.items()}
        for step in range(min(len(w) for w, _, _ in curves.values())):
            mlflow.log_metrics({f"{name}.block_score": int(w[step] - l[step])
                                for name, (w, _, l) in curves.items()}, step=step)

        fig = plot_results(results, block=BLOCK)
        mlflow.log_figure(fig, "results.png")
        plt.close(fig)

        learned = {name: a.state_dict() for name, a in agents.items()}
        if any(learned.values()):
            mlflow.log_dict(learned, "state_dicts.json")
