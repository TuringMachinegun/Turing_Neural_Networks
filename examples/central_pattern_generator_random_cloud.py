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
import numpy as np

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
init_state = ge_alpha.encode_sequence(["g", "1"])
init_tape = ge_beta.encode_sequence(list("1"))


# run R-ANN with random cloud of initial conditions
def rand_cloud_run(xs, ys, n_iter):
    init_conds = zip(xs, ys)
    mean_acts = np.zeros((xs.size, n_iter))

    for i, ic in enumerate(init_conds):
        x = ic[0]
        y = ic[1]
        tm_nn.set_init_cond(x, y)

        for j in range(n_iter):
            MCL_acts = np.concatenate((tm_nn.MCLx.activation, tm_nn.MCLy.activation))
            tm_nn.run_net()
            all_acts = np.concatenate(
                (
                    MCL_acts,
                    tm_nn.BSLbx.activation,
                    tm_nn.BSLby.activation,
                    tm_nn.LTL.activation,
                )
            )
            mean_acts[i, j] = np.mean(all_acts)
    return np.mean(mean_acts, axis=0), np.std(mean_acts, axis=0)


n_iter = 32
n_init_conds = 100

w_rd_x = np.random.uniform(0.0, 0.5, size=n_init_conds)
w_rd_y = np.random.uniform(0, 0.25, size=n_init_conds)
walk_means, walk_std = rand_cloud_run(w_rd_x, w_rd_y, n_iter)

g_rd_x = np.random.uniform(0.5, 1, size=n_init_conds)
g_rd_y = np.random.uniform(0, 0.25, size=n_init_conds)
gall_means, gall_std = rand_cloud_run(g_rd_x, g_rd_y, n_iter)

# Plot syntetic ERPs
plt.ion()
plt.figure(figsize=[8, 4])
plt.style.use("ggplot")

plt.fill_between(
    range(n_iter), walk_means - walk_std, walk_means + walk_std, color="cornflowerblue"
)
plt.plot(range(n_iter), walk_means, label="Walk gait", color="lightblue", lw=2)

plt.fill_between(
    range(n_iter), gall_means - gall_std, gall_means + gall_std, color="lightpink"
)
plt.plot(range(n_iter), gall_means, label="Gallop gait", color="red", lw=2)


plt.ylim([0, 0.2])
plt.xlim([0, n_iter - 1])
plt.xlabel("Time step")
plt.ylabel("Mean network activation")
plt.legend()
plt.tight_layout()
