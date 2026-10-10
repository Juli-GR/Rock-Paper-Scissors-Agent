"""Gráficos de cómo evoluciona un agente a lo largo de una partida."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"gana": "#0ca30c", "empata": "#a3a3a3", "pierde": "#d03b3b"}
INK, GRID = "#2b2b2b", "#e5e5e5"


def blocks(rewards: list[int], size: int):
    """Agrupa las rewards en bloques de `size` rondas -> (wins, ties, losses) por bloque."""
    r = np.array(rewards[: len(rewards) // size * size]).reshape(-1, size)
    return (r == 1).sum(1), (r == 0).sum(1), (r == -1).sum(1)


def plot_results(results: dict[str, list[int]], block=20, cols=4):
    """Un panel por oponente, en una grilla de `cols` columnas.

    Arriba: score por bloque. Abajo: de qué está hecho ese score (W/T/L).
    """
    n = len(results)
    cols = min(n, cols)
    rows = -(-n // cols)  # división redondeando hacia arriba
    fig = plt.figure(figsize=(4 * cols, 4.5 * rows + 0.5), layout="constrained")
    panels = np.atleast_1d(fig.subfigures(rows, cols)).ravel()
    for panel in panels[n:]:
        panel.set_visible(False)

    for i, (panel, (name, rewards)) in enumerate(zip(panels, results.items())):
        w, t, l = blocks(rewards, block)
        x = np.arange(len(w)) * block + block / 2  # centro de cada bloque, en rondas
        top, bottom = panel.subplots(2, 1, sharex=True)

        top.set_title(name)
        top.plot(x, w - l, color=INK, lw=2)
        top.axhline(0, color=INK, lw=0.8, alpha=0.4)
        top.set_ylim(-block, block)

        base = np.zeros_like(w)
        for (label, color), counts in zip(COLORS.items(), (w, t, l)):
            bottom.bar(x, counts, bottom=base, width=block, color=color,
                       edgecolor="white", linewidth=1, label=label)
            base = base + counts
        bottom.set_ylim(0, block)
        bottom.set_xlabel("ronda")
        if i % cols == 0:
            top.set_ylabel(f"score cada {block}")
            bottom.set_ylabel(f"rondas (de {block})")

        for ax in (top, bottom):
            ax.grid(axis="y", color=GRID)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = bottom.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    return fig
