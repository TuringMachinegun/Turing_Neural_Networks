__author__ = 'Giovanni Sirio Carmantini'

"""In this file we reproduce the simple parser from beim Graben, P.,
 & Potthast, R. (2014). Universal neural field computation. In Neural
 Fields (pp. 299-318).

First, the Context Free Grammar is used to create a Generalized Shift,
an NDA simulating the GS is then created, then the NDA-simulating
R-ANN is constructed from the NDA.

Finally, the R-ANN dynamics is simulated from given initial conditions
and visualized.

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
from matplotlib.patches import Rectangle
from plot_symbologram import plot_sym
import numpy as np
from time import sleep

# CFG description
input_symbols = ["NP", "V"]
stack_symbols = ["NP", "V", "VP", "S"]
parser_descr = {"S": ["NP", "VP"],
                "VP": ["V", "NP"],
                }

# Godel Encoders
ge_s = symdyn.GodelEncoder(stack_symbols)
ge_i = symdyn.GodelEncoder(input_symbols)

# CFG -> GS
cfg_gs = symdyn.SimpleCFGeneralizedShift(stack_symbols,
                                         input_symbols,
                                         parser_descr)
# GS -> NDA
nda = symdyn.NonlinearDynamicalAutomaton(cfg_gs, ge_s, ge_i)

# NDA -> R-ANN
cfg_nn = neuraltm.NeuralTM(nda,
                           cylinder_sets=True)

# initial conditions
init_stack = ge_s.encode_cylinder("S")
init_input = ge_i.encode_cylinder(["NP", "V", "NP"])

# simulate NDA and R-ANN dynamics
nda_states = nda.iterate(init_stack, init_input, 6)
cfg_states = cfg_nn.run_net(init_x=init_stack, init_y=init_input,
                            n_iterations=6)

# and plot
plt.ion()
plt.style.use("ggplot")
fig = plt.figure(figsize=[5, 5])
ax2 = plt.axes(aspect="equal")
ax2.axis([0, 1, 0, 1])

plot_sym(ax2, stack_symbols, input_symbols, ge_s, ge_i, TM=False)

x_states, y_states = zip(*cfg_states)
ax2.set_xlabel("$c_x$ activation", size=15)
ax2.set_ylabel("$c_y$ activation", size=15)
plt.tight_layout()

states_old = None
for i, cfg_state in enumerate(cfg_states):

    if np.array_equal(cfg_state, states_old):
        break
    x, y = cfg_state[0][0], cfg_state[1][0]
    w_x = cfg_state[0][1] - x
    w_y = cfg_state[1][1] - y

    ax2.add_patch(Rectangle((x, y), w_x, w_y, facecolor="blue", zorder=i))

    ax2.annotate("{}".format(i + 1),
                 xy=(cfg_state[0][0], cfg_state[1][0]),
                 xytext= (x + w_x / 2.0, y + w_y / 2.0),
                 size=15, zorder=i)

    states_old = cfg_state
    plt.draw()
    plt.pause(1)

print "total number of neurons: {}".format(cfg_nn.LTL.n_units +
                                           cfg_nn.BSLbx.n_units +
                                           cfg_nn.BSLby.n_units +
                                           cfg_nn.MCLx.n_units +
                                           cfg_nn.MCLy.n_units)
