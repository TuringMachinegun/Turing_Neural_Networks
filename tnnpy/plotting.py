import itertools as itt
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from tnnpy.symdyn import GodelEncoder, CompactGodelEncoder


def plot_symbologram(
        ax: plt.Axes,
        alpha_symbols: List[str],
        beta_symbols: List[str],
        ge_alpha: GodelEncoder,
        ge_beta: Union[GodelEncoder, CompactGodelEncoder],
        TM: bool = True):
    """Plot symbologram"""
    if TM:
        state_ticks_xpos = [
            ge_alpha.ge_q.encode_sequence(state) for state in alpha_symbols
        ] + [1.0]
    else:
        state_ticks_xpos = [
            ge_alpha.encode_sequence(state) for state in alpha_symbols
        ] + [1.0]
    state_ticks_labels = ["" for _ in state_ticks_xpos]
    state_labels_xpos = [x + state_ticks_xpos[1] / 2.0 for x in state_ticks_xpos[:-1]]
    state_labels = [f"${x}$" for x in alpha_symbols]
    maj_xtick_pos, maj_xtick_labels = zip(
        *sorted(
            list(zip(state_ticks_xpos, state_ticks_labels)) + list(zip(state_labels_xpos, state_labels))
        )
    )
    ax.set_xticks(maj_xtick_pos, maj_xtick_labels, size=20)
    ax.xaxis.set_tick_params(pad=20, length=10, which="major")

    if TM:
        sym_ticks_xpos = [
            ge_alpha.encode_sequence([x[0], x[1]])
            for x in itt.product(alpha_symbols, beta_symbols)
        ]
        sym_ticks_labels = ["" for _ in sym_ticks_xpos]
        sym_labels_xpos = [x + sym_ticks_xpos[1] / 2.0 for x in sym_ticks_xpos[::-1]]
        sym_labels_x = [f"${x}$" for x in beta_symbols * len(alpha_symbols)]

        min_xtick_pos, min_xtick_labels = zip(
            *sorted(
                list(zip(sym_ticks_xpos, sym_ticks_labels)) +
                list(zip(sym_labels_xpos, sym_labels_x)) +
                list(zip(maj_xtick_pos, ["" for _ in maj_xtick_pos]))
            )
        )
        ax.set_xticks(min_xtick_pos, min_xtick_labels, size=20, minor=True)
        ax.xaxis.set_tick_params(pad=20, length=10, which="major")

    for i, x in enumerate(ax.xaxis.get_major_ticks()):
        if i % 2 == 1:
            if TM:
                x.tick1line.set_alpha(.2)
                x.tick2line.set_alpha(.2)
            else:
                x.tick1line.set_visible(False)
                x.tick2line.set_visible(False)

    for i, x in enumerate(ax.xaxis.get_minor_ticks()):
        x.tick1line.set_visible(False)
        x.tick2line.set_visible(False)

    for xmaj in ax.xaxis.get_majorticklocs()[::2]:
        ax.axvline(x=xmaj, lw=1.2, c="white")
    if TM:
        for xmin in ax.xaxis.get_majorticklocs()[1::2]:
            ax.axvline(x=xmin, lw=1.2, ls="dotted", c="white")

    sym_ticks_ypos = [ge_beta.encode_sequence([x]) for x in beta_symbols] + [1.0]
    sym_labels_ypos = [x + sym_ticks_ypos[1] / 2.0 for x in sym_ticks_ypos[:-1]]
    sym_labels_y = [f"${x}$" for x in beta_symbols]
    ax.set_yticks(sym_labels_ypos, sym_labels_y, size=20, minor=True)
    ax.set_yticks(sym_ticks_ypos, [x for x in itt.repeat("", len(sym_ticks_ypos))], size=20)
    ax.yaxis.set_tick_params(pad=20, length=10)

    for i, x in enumerate(ax.yaxis.get_minor_ticks()):
        x.tick1line.set_visible(False)
        x.tick2line.set_visible(False)

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, lw=1.2, c="white")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.grid(False)

    return ax


def plot_cylinders(ax: plt.Axes, cylinders_2d: List[Tuple[np.ndarray, np.ndarray]]):
    """Plot cylinder sets on ax."""
    for cylinder in cylinders_2d:
        x_cyl, y_cyl = cylinder
        x, w_x = x_cyl[0], x_cyl[1] - x_cyl[0]
        y, w_y = y_cyl[0], y_cyl[1] - y_cyl[0]
        rect = Rectangle((x, y), w_x, w_y, facecolor="orange", edgecolor="black", zorder=5)
        ax.add_patch(rect)
