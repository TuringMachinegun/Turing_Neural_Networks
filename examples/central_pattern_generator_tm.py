__author__ = "Giovanni Sirio Carmantini"

"""In this file, a R-ANN is constructed from a Turing Machine which
dynamics reproduce two gait pattern sequences, depending on the
machine control state.

A Generalized Shift is first created from the TM description.
Subsequently, an NDA simulating the GS dynamics is created.
Then, a R-ANN is constructed from the NDA.

The dynamics of the R-ANN is simulated, while manipulating
the activation of the c_x neuron, and the two gait patterns are
produced. Results are visualized.

"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from tnnpy import GodelEncoder, CompactGodelEncoder, TMGeneralizedShift, NonlinearDynamicalAutomaton, NeuralTM

# Turing Machine description (latex syntax for typesetting in plot)
tape_symbols = ["1", "2", "3", "4"]
states = ["w", "g"]
tm_descr = {
    ("w", "1"): ("w", "3", "S"),
    ("w", "2"): ("w", "4", "S"),
    ("w", "3"): ("w", "2", "S"),
    ("w", "4"): ("w", "1", "S"),
    ("g", "1"): ("g", "2", "S"),
    ("g", "2"): ("g", "3", "S"),
    ("g", "3"): ("g", "4", "S"),
    ("g", "4"): ("g", "1", "S"),
}

# create encoders for states and tape symbols
ge_q = GodelEncoder(states)
ge_s = GodelEncoder(tape_symbols)

# and from the simple encoders, create the actual encoders for the
# alpha and beta subsequences
ge_alpha = CompactGodelEncoder(ge_q, ge_s)
ge_beta = ge_s

# create Generalized Shift from machine description...
tm_gs = TMGeneralizedShift(states, tape_symbols, tm_descr)

# ...then NDA from the Generalized Shift and encoders...
nda = NonlinearDynamicalAutomaton(tm_gs, ge_alpha, ge_beta)

# ... and finally the R-ANN simulating the TM from the NDA
tm_nn = NeuralTM(nda)

# set initial conditions for the computation
init_state = ge_alpha.encode_sequence(["g", "1"])
init_tape = ge_beta.encode_sequence(list("1"))

# run R-ANN
inp = np.zeros(16)  # set input for each time step
inp[:8] = 0.3
inp[8:] = 0.7

tm_nn.MCLx.activation[:] = 0
tm_nn.MCLy.activation[:] = init_tape

iterations = inp.size

MCL_n = tm_nn.MCLx.n_units + tm_nn.MCLy.n_units
BSL_n = tm_nn.BSLbx.n_units + tm_nn.BSLby.n_units
LTL_n = tm_nn.LTL.n_units

ld = OrderedDict(
    [
        ("LTL", {"acts": np.zeros((LTL_n, iterations)), "n": LTL_n}),
        ("BSL", {"acts": np.zeros((BSL_n, iterations)), "n": BSL_n}),
        ("MCL", {"acts": np.zeros((MCL_n, iterations)), "n": MCL_n}),
    ]
)

# run
for i, input_state in enumerate(inp):
    tm_nn.MCLx.activation[:] = input_state
    ld["MCL"]["acts"][:, i] = np.concatenate(
        (tm_nn.MCLx.activation, tm_nn.MCLy.activation)
    )
    tm_nn.run_net()
    ld["BSL"]["acts"][:, i] = np.concatenate(
        (tm_nn.BSLbx.activation, tm_nn.BSLby.activation)
    )
    ld["LTL"]["acts"][:, i] = tm_nn.LTL.activation


# ...and plot
timeseries_fig = plt.figure()
gs = gridspec.GridSpec(
    nrows=6,
    ncols=2,
    width_ratios=[14, 1],
    height_ratios=[20] + [ld[l]["n"] for l in ld] + [3, 20],
)

for i, k in enumerate(ld, 1):
    ld[k]["ax"] = plt.subplot(gs[i, 0])
    n = float(ld[k]["acts"].shape[0])
    ld[k]["plot"] = ld[k]["ax"].pcolor(ld[k]["acts"], cmap="OrRd")
    ld[k]["ax"].set_ylim([0, n])
    ld[k]["ax"].set_yticks([n / 2])
    ld[k]["ax"].set_yticklabels([k + " units"])
    ld[k]["ax"].set_xticks(range(iterations))
    ld[k]["ax"].set_xticklabels([])
    for tick in ld[k]["ax"].yaxis.get_major_ticks():
        tick.tick1On = tick.tick2On = False
    for tick in ld[k]["ax"].xaxis.get_major_ticks():
        tick.tick1On = tick.tick2On = False
    plt.grid(axis="x")

cbar_ax = plt.subplot(gs[1:4, 1])
cbar_ax.set_xticks([])
ld[k]["plot"].set_clim(vmin=0, vmax=1)
cbar = plt.colorbar(ld[k]["plot"], cbar_ax)
cbar.solids.set_edgecolor("face")
cbar.solids.set_rasterized(True)
cbar_ax.set_ylabel("Activation")

inp_ax = plt.subplot(gs[5, 0])
inp_axr = inp_ax.twinx()
inp_ax.set_xlim([0, iterations])
inp_axr.set_xlim([0, iterations])
inp_ax.bar(range(inp.size), inp, width=1, edgecolor="none", facecolor="black")
inp_ax.set_yticks([0.5])
inp_ax.set_yticklabels(["$c_x$ activation"])
inp_ax.set_ylim([0, 1])
inp_axr.set_ylim([0, 1])
inp_axr.set_yticks([0, 1])
inp_ax.set_xticks(range(iterations))
inp_ax.set_xticklabels([])
inp_ax.set_xticks(np.array(range(iterations)) + 0.5, minor=True)
inp_ax.set_xticklabels(range(iterations), ha="center", minor=True)
inp_ax.grid(axis="x", which="major")
inp_ax.set_xlabel("Time step")
inp_ax.arrow(
    float(iterations) / 2,
    1.1,
    0,
    0.8,
    fc="black",
    ec="black",
    width=0.5,
    head_width=1,
    head_length=0.2,
    clip_on=False,
    length_includes_head=True,
)
for tick in inp_ax.yaxis.get_major_ticks():
    tick.tick1On = tick.tick2On = False
for tick in inp_ax.xaxis.get_minor_ticks():
    tick.tick1On = tick.tick2On = False

gait_ax = plt.subplot(gs[0, 0])
gait_ax.set_xticks(range(9))
gait_ax.set_xticklabels([])
gait_ax.set_yticks([])
gait_ax.set_yticklabels([])
gait_ax.grid(axis="x", linestyle="-")
for tick in gait_ax.xaxis.get_major_ticks():
    tick.tick1On = tick.tick2On = False
plt.tight_layout()

# Plot also syntetic ERPs
synth_fig = plt.figure()
s_ax = synth_fig.add_subplot(111)
plt.style.use("ggplot")

all_acts = np.concatenate(
    (ld["MCL"]["acts"], ld["BSL"]["acts"], ld["LTL"]["acts"]), axis=0
)
walk_acts = np.mean(all_acts, axis=0)[:8]
gallop_acts = np.mean(all_acts, axis=0)[8:]

s_ax.plot(range(8), walk_acts, label="Walk gait", color="blue", lw=2)
s_ax.plot(range(8), gallop_acts, label="Gallop gait", color="red", lw=2)
s_ax.axis([0, 7, 0, 0.2])
s_ax.set_xlabel("Time step")
s_ax.set_ylabel("Mean network activation")
s_ax.legend()
plt.show()