import itertools as itt


def plot_sym(ax, alpha_symbols, beta_symbols,
             ge_alpha, ge_beta, TM=True):

    if TM:
        state_ticks_xpos = [
            ge_alpha.ge_q.encode_sequence(state) for state in alpha_symbols] + [1.0]
    else:
        state_ticks_xpos = [
            ge_alpha.encode_sequence(state) for state in alpha_symbols] + [1.0]

    state_labels_xpos = [
        x + state_ticks_xpos[1] / 2.0 for x in state_ticks_xpos]
    ax.set_xticks(sorted(state_ticks_xpos + state_labels_xpos))
    state_labels = [""] * len(ax.xaxis.get_major_ticks()[:-1])
    state_labels[1::2] = ['$' + x + '$' for x in alpha_symbols]
    ax.set_xticklabels(state_labels, size=20)

    if TM:
        sym_ticks_xpos = [ge_alpha.encode_sequence([x[0], x[1]])
                          for x in itt.product(alpha_symbols, beta_symbols)]
        sym_labels_xpos = [x + sym_ticks_xpos[1] / 2.0 for x in sym_ticks_xpos]
        ax.set_xticks(sorted(sym_ticks_xpos + sym_labels_xpos), minor=True)
        sym_labels_x = [""] * len(ax.xaxis.get_minor_ticks())
        sym_labels_x[1::2] = [
            '$' + x + '$'for x in beta_symbols * len(alpha_symbols)]
        ax.set_xticklabels(sym_labels_x, minor=True)
        ax.xaxis.set_tick_params(pad=20, length=10)

    for i, x in enumerate(ax.xaxis.get_major_ticks()):
        if i % 2 == 1:
            x.tick1On = False
            x.tick2On = False

    for i, x in enumerate(ax.xaxis.get_minor_ticks()):
        if i % 2 == 1:
            x.tick1On = False
            x.tick2On = False

    for xmaj in ax.xaxis.get_majorticklocs()[2::2]:
        ax.axvline(x=xmaj, lw=1.2, c='white')
    for xmin in ax.xaxis.get_minorticklocs()[2::4]:
        ax.axvline(x=xmin, lw=1.2, ls="dotted", c='white')

    sym_ticks_ypos = [
        ge_beta.encode_sequence([x]) for x in beta_symbols] + [1.0]
    sym_labels_ypos = [x + sym_ticks_ypos[1] / 2.0 for x in sym_ticks_ypos]
    ax.set_yticks(sorted(sym_ticks_ypos + sym_labels_ypos))
    sym_labels_y = [""] * len(ax.yaxis.get_major_ticks()[:-1])
    sym_labels_y[1::2] = ['$' + x + '$'for x in beta_symbols]

    ax.set_yticklabels(sym_labels_y, size=20)
    ax.yaxis.set_tick_params(pad=20, length=10)

    for i, x in enumerate(ax.yaxis.get_major_ticks()):
        if i % 2 == 1:
            x.tick1On = False
            x.tick2On = False

    for ymaj in ax.yaxis.get_majorticklocs()[2::2][:-1]:
        ax.axhline(y=ymaj, lw=1.2, c='white')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.grid(False)

    return ax
