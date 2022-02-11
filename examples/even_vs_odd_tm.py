__author__ = "Giovanni Sirio Carmantini"

"""In this file, a R-ANN is constructed from a sample Turing Machine.

Specifically, the TM decides if an input unary string is composed by
an even number of 1's.

A Generalized Shift is first created from the TM description.
Subsequently, an NDA simulating the GS dynamics is created.
Then, a R-ANN is constructed from the NDA.

Finally, the dynamics of the R-ANN is simulated from initial
conditions and visualized.

"""
import os.path
import sys
import inspect

curr_file_path = os.path.realpath(inspect.getfile(inspect.currentframe()))
curr_dir_path = os.path.dirname(curr_file_path)
parent_dir = os.path.join(curr_dir_path, os.path.pardir)
sys.path.append(parent_dir)

import symdyn
import neuraltm
import matplotlib.pyplot as plt
from plot_symbologram import plot_sym

# Turing Machine description (latex syntax for typesetting in plot)
tape_symbols = ["\\sqcup", "1"]
states = ["q_{acc}", "q_{rej}", "q_{even}", "q_{odd}"]
tm_descr = {
    ("q_{even}", "1"): ("q_{odd}", "1", "R"),
    ("q_{even}", "\\sqcup"): ("q_{acc}", "\\sqcup", "L"),
    ("q_{odd}", "1"): ("q_{even}", "1", "R"),
    ("q_{odd}", "\\sqcup"): ("q_{rej}", "\\sqcup", "L"),
    ("q_{acc}", "1"): ("q_{acc}", "1", "S"),
    ("q_{acc}", "\\sqcup"): ("q_{acc}", "\\sqcup", "S"),
    ("q_{rej}", "1"): ("q_{rej}", "1", "S"),
    ("q_{rej}", "\\sqcup"): ("q_{rej}", "\\sqcup", "S"),
}

# create encoders for states and tape symbols
ge_q = symdyn.GodelEncoder(states)
ge_s = symdyn.GodelEncoder(tape_symbols)

# and from the simple encoders, create the actual encoders for the
# alpha and beta subsequences
ge_alpha = symdyn.compactGodelEncoder(ge_q, ge_s)
ge_beta = ge_s

# create Generalized Shift from machine description...
tm_gs = symdyn.TMGeneralizedShift(states, tape_symbols, tm_descr)

# ...then NDA from the Generalized Shift and encoders...
nda = symdyn.NonlinearDynamicalAutomaton(tm_gs, ge_alpha, ge_beta)

# ... and finally the R-ANN simulating the TM from the NDA
tm_nn = neuraltm.NeuralTM(nda)

# set initial conditions for the computation
init_alpha = ge_alpha.encode_sequence(["q_{even}", "\\sqcup"])
init_beta_acc = ge_beta.encode_sequence(list("1111"))
init_beta_rej = ge_beta.encode_sequence(list("111"))

# run R-ANN
tm_nn_configs_acc = tm_nn.run_net(
    init_x=init_alpha, init_y=init_beta_acc, n_iterations=10
)
tm_nn_configs_rej = tm_nn.run_net(
    init_x=init_alpha, init_y=init_beta_rej, n_iterations=10
)

# plot results
plt.ion()
plt.style.use("ggplot")
fig = plt.figure(figsize=[10, 5])

axl = fig.add_subplot(121, aspect="equal")
axl.axis([0, 1, 0, 1])

axr = fig.add_subplot(122, aspect="equal")
axr.axis([0, 1, 0, 1])


def plot_dynamics(axis, tm_nn_configs):
    plot_sym(axis, states, tape_symbols, ge_alpha, ge_beta, TM=True)

    x_states, y_states = zip(*tm_nn_configs)
    axis.plot(x_states, y_states, linestyle="", marker=".", ms=10)

    states_old = None
    for i, tm_nn_config in enumerate(tm_nn_configs):

        if tm_nn_config == states_old:
            break

        axis.annotate(
            "{}".format(i + 1),
            xy=tm_nn_config,
            xytext=(tm_nn_config[0] - 0.03, tm_nn_config[1] + 0.01),
            size=15,
        )

        states_old = tm_nn_config

    axis.set_xlabel("$c_x$ activation", size=15)
    axis.set_ylabel("$c_y$ activation", size=15)


plot_dynamics(axl, tm_nn_configs_acc)
plot_dynamics(axr, tm_nn_configs_rej)

axr.set_ylabel("")
axr.set_yticklabels("")
for t in axl.yaxis.get_major_ticks():
    t.tick2On = False
for t in axr.yaxis.get_major_ticks():
    t.tick1On = False

plt.tight_layout()

print(
    "total number of neurons: {}".format(
        tm_nn.LTL.n_units
        + tm_nn.BSLbx.n_units
        + tm_nn.BSLby.n_units
        + tm_nn.MCLx.n_units
        + tm_nn.MCLy.n_units
    )
)
plt.show()
