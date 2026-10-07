"""Gráficos de cómo evoluciona un agente a lo largo de una partida."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"gana": "#0ca30c", "empata": "#a3a3a3", "pierde": "#d03b3b"}
INK, GRID = "#2b2b2b", "#e5e5e5"


def blocks(rewards: list[int], size: int):
    """Agrupa las rewards en bloques de `size` rondas -> (wins, ties, losses) por bloque."""
    r = np.array(rewards[: len(rewards) // size * size]).reshape(-1, size)
    return (r == 1).sum(1), (r == 0).sum(1), (r == -1).sum(1)


def plot_results(results: dict[str, list[int]], block=20):
    """Arriba: score por bloque. Abajo: de qué está hecho ese score (W/T/L)."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6), sharex=True, sharey="row", squeeze=False)
    for col, (name, rewards) in enumerate(results.items()):
        w, t, l = blocks(rewards, block)
        x = np.arange(len(w)) * block + block / 2  # centro de cada bloque, en rondas
        top, bottom = axes[:, col]

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

        for ax in (top, bottom):
            ax.grid(axis="y", color=GRID)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel(f"score cada {block} rondas")
    axes[1, 0].set_ylabel(f"rondas (de {block})")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
