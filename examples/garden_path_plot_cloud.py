__author__ = 'Giovanni Sirio Carmantini'
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import tnnpy.symdyn as sd
from tnnpy import neuraltm

# Parsers and repair shifts
input_syms = [" ", "s", "o"]
parse_syms = [" ", "s", "o", "S"]
A_delta = {"S": ["s", "o"], " ": [" "]}  # Parser for subject-object sentences
B_delta = {"S": ["o", "s"], " ": [" "]}  # Parser for object-subject sentences
R_delta = {("s", "o"): ("o", "s")}  # Repair Generalized Shift

pa_ge_s = sd.GodelEncoder(parse_syms, force_power_of_two=True)
pa_ge_i = sd.GodelEncoder(input_syms, force_power_of_two=True)

A_cfg_gs = sd.SimpleCFGeneralizedShift(parse_syms, input_syms, A_delta)
B_cfg_gs = sd.SimpleCFGeneralizedShift(parse_syms, input_syms, B_delta)
R_gs = sd.RepairGeneralizedShift(parse_syms, input_syms, R_delta)

A_nda = sd.NonlinearDynamicalAutomaton(A_cfg_gs, pa_ge_s, pa_ge_i)
B_nda = sd.NonlinearDynamicalAutomaton(B_cfg_gs, pa_ge_s, pa_ge_i)
R_nda = sd.NonlinearDynamicalAutomaton(R_gs, pa_ge_s, pa_ge_i)

A_nn = neuraltm.NeuralTM(A_nda, cylinder_sets=False, label="A")
B_nn = neuraltm.NeuralTM(B_nda, cylinder_sets=False, label="B")
R_nn = neuraltm.NeuralTM(R_nda, cylinder_sets=False, label="R")

# Memory Encoder
M_q_syms = ["idle", "parsing", "error"]
M_i_syms = parse_syms
M_s_syms = parse_syms

M_delta = {}

nor_trans = {("parsing", i, s): ("parsing", i)
             for i, s in product(M_i_syms, M_s_syms) if i != s}
nor_trans.update({("error", i, s): ("parsing", i) for i, s in
                  product(M_i_syms, M_s_syms) if i != s})

err_trans = {("parsing", i, s): ("error", i) for i, s in
             product(M_i_syms, M_s_syms)
             if (i == s and not (i == " " and s == " "))}
err_trans.update({("error", i, s): ("error", i) for i, s in
                  product(M_i_syms, M_s_syms) if i == s})

par_trans = {("parsing", " ", " "): ("idle", " "),
             ("idle", " ", " "): ("idle", " ")}
par_trans.update({("idle", i, s): ("parsing", i)
                 for i, s in product(M_i_syms, M_s_syms)
                  if not (i == " " and s == " ")})

M_delta.update(nor_trans)
M_delta.update(err_trans)
M_delta.update(par_trans)

M_ge_q = sd.GodelEncoder(M_q_syms, force_power_of_two=True)
M_ge_alpha = sd.CompactGodelEncoder(M_ge_q, pa_ge_s)
M_ge_beta = pa_ge_s

M_gs = sd.PDAGeneralizedShift(M_q_syms, M_i_syms, M_s_syms, M_delta)
M_nda = sd.NonlinearDynamicalAutomaton(M_gs, M_ge_alpha, M_ge_beta)
M_nn = neuraltm.NeuralTM(M_nda, cylinder_sets=False, label="M")

# Strategy selector
S_q_syms = ["A", "B", "R"]
S_i_syms = M_q_syms

S_delta = {
    ("A", "parsing"): ("A"),
    ("A", "error"): ("R"),
    ("A", "idle"): ("A"),
    ("R", "parsing"): ("B"),
    ("R", "error"): ("B"),
    ("R", "idle"): ("A"),
    ("B", "parsing"): ("B"),
    ("B", "error"): ("B"),
    ("B", "idle"): ("A")
}

S_ge_q = sd.GodelEncoder(S_q_syms, force_power_of_two=True)
S_ge_i = sd.GodelEncoder(S_i_syms, force_power_of_two=True)
S_gs = sd.FSMGeneralizedShift(S_q_syms, S_i_syms, S_delta)
S_nda = sd.NonlinearDynamicalAutomaton(S_gs, S_ge_q, S_ge_i)
S_nn = neuraltm.NeuralTM(S_nda, cylinder_sets=False, label="S")


# ------------------------------------#
# Construct complete neural network   #
# ------------------------------------#

def subst_conn(connection, layer_old, layer_new):
    """
    Helper function to substitute layer connections
    """
    layer, conn_mat = connection
    if layer == layer_old:
        layer = layer_new
    return layer, conn_mat

# Connect the A_nn MCL (used as the MCL for the entire parsing subnetwork)
# to the other parsing networks
for l_name, l in B_nn.layers.items():
    l.connections = list(map(lambda c: subst_conn(c, B_nn.MCLx, A_nn.MCLx),
                        l.connections))
    l.connections = list(map(lambda c: subst_conn(c, B_nn.MCLy, A_nn.MCLy),
                        l.connections))
for l in R_nn.layers.values():
    l.connections = list(map(lambda c: subst_conn(c, R_nn.MCLx, A_nn.MCLx),
                        l.connections))
    l.connections = list(map(lambda c: subst_conn(c, R_nn.MCLy, A_nn.MCLy),
                        l.connections))

# Connect the other networks to the A_nn MCL
A_nn.MCLx.connections += B_nn.MCLx.connections + R_nn.MCLx.connections
A_nn.MCLy.connections += B_nn.MCLy.connections + R_nn.MCLy.connections
# Substitute MCL in network objects so that they get updated
B_nn.MCLx, B_nn.MCLy = A_nn.MCLx, A_nn.MCLy
R_nn.MCLx, R_nn.MCLy = A_nn.MCLx, A_nn.MCLy

# Connect the A_nn MCLx (the parse) to the memory network
for l_name, l in M_nn.layers.items():
    l.connections = list(map(lambda c: subst_conn(c, M_nn.MCLy, A_nn.MCLx),
                        l.connections))
M_nn.MCLy = A_nn.MCLx

# Connect the M_nn MCLx (the stack) to the strategy selector
for l_name, l in S_nn.layers.items():
    l.connections = list(map(lambda c: subst_conn(c, S_nn.MCLy, M_nn.MCLx),
                        l.connections))
S_nn.MCLy = M_nn.MCLx


# Connect the S_nn branch selection layer to LTL of strategies
h_const = 8.0
S_connmats = {}
S_connmats["A"] = {"LTL": np.zeros(
    (S_nn.BSLbx.activation.size, A_nn.LTL.activation.size)),
    "BSLbx": np.zeros(
        (S_nn.BSLbx.activation.size, A_nn.BSLbx.activation.size)),
    "BSLby": np.zeros(
        (S_nn.BSLbx.activation.size, A_nn.BSLby.activation.size))}
S_connmats["B"] = {"LTL": np.zeros(
    (S_nn.BSLbx.activation.size, B_nn.LTL.activation.size)),
    "BSLbx": np.zeros(
        (S_nn.BSLbx.activation.size, B_nn.BSLbx.activation.size)),
    "BSLby": np.zeros(
        (S_nn.BSLbx.activation.size, B_nn.BSLby.activation.size))}
S_connmats["R"] = {"LTL": np.zeros(
    (S_nn.BSLbx.activation.size, R_nn.LTL.activation.size)),
    "BSLbx": np.zeros(
        (S_nn.BSLbx.activation.size, R_nn.BSLbx.activation.size)),
    "BSLby": np.zeros(
        (S_nn.BSLbx.activation.size, R_nn.BSLby.activation.size))}

for s in ["A", "B", "R"]:
    cell_n = S_ge_q.gamma[s]
    S_connmats[s]["LTL"][cell_n, :] = h_const
    S_connmats[s]["BSLbx"][cell_n] = h_const
    S_connmats[s]["BSLby"][cell_n] = h_const
    if cell_n < 2:
        S_connmats[s]["LTL"][cell_n + 1, :] = - h_const
        S_connmats[s]["BSLbx"][cell_n + 1] = - h_const
        S_connmats[s]["BSLby"][cell_n + 1] = - h_const

A_nn.LTL.connections.append((S_nn.BSLbx, S_connmats["A"]["LTL"]))
A_nn.BSLbx.connections.append((S_nn.BSLbx, S_connmats["A"]["BSLbx"]))
A_nn.BSLby.connections.append((S_nn.BSLbx, S_connmats["A"]["BSLby"]))
B_nn.LTL.connections.append((S_nn.BSLbx, S_connmats["B"]["LTL"]))
B_nn.BSLbx.connections.append((S_nn.BSLbx, S_connmats["B"]["BSLbx"]))
B_nn.BSLby.connections.append((S_nn.BSLbx, S_connmats["B"]["BSLby"]))
R_nn.LTL.connections.append((S_nn.BSLbx, S_connmats["R"]["LTL"]))
R_nn.BSLbx.connections.append((S_nn.BSLbx, S_connmats["R"]["BSLbx"]))
R_nn.BSLby.connections.append((S_nn.BSLbx, S_connmats["R"]["BSLby"]))

A_nn.LTL.biases += -h_const
A_nn.BSLbx.biases += -h_const
A_nn.BSLby.biases += -h_const
B_nn.LTL.biases += -h_const
B_nn.BSLbx.biases += -h_const
B_nn.BSLby.biases += -h_const
R_nn.LTL.biases += -h_const
R_nn.BSLbx.biases += -h_const
R_nn.BSLby.biases += -h_const


def run_whole_aut(input_parse_external, n_iter):

    init_p_input = [" "]
    init_p_parse = [" "]
    init_m_input = init_p_parse
    init_m_state = ["idle", " "]
    init_s_input = init_m_state
    init_s_state = ["A"]

    tapes = [(init_s_state, init_m_state, init_p_parse, init_p_input)]
    parse_before_and_after = []

    presentation_i = input_parse_external.keys()

    for i in range(n_iter):
        stimulus_presentation = i in presentation_i

        curr_s_state, curr_m_state, curr_p_parse, curr_p_input = tapes[-1]
        if stimulus_presentation:
            curr_p_parse, curr_p_input = input_parse_external[i]

        parse_before = curr_p_parse

        if curr_s_state[0] == "A":
            curr_p_parse, curr_p_input = A_cfg_gs.iterate(
                curr_p_parse, curr_p_input)[1]
        elif curr_s_state[0] == "B":
            curr_p_parse, curr_p_input = B_cfg_gs.iterate(
                curr_p_parse, curr_p_input)[1]
        elif curr_s_state[0] == "R":
            curr_p_parse, curr_p_input = R_gs.iterate(
                curr_p_parse, curr_p_input)[1]

        if not curr_p_input:
            curr_p_input = [" "]

        parse_after = curr_p_parse

        parse_before_and_after.append((parse_before, parse_after))

        curr_m_state, _ = M_gs.iterate(curr_m_state, curr_p_parse)[1]
        curr_s_state, _ = S_gs.iterate(curr_s_state, curr_m_state)[1]
        tapes.append((curr_s_state, curr_m_state, curr_p_parse, curr_p_input))
    return tapes, parse_before_and_after


def run_whole_net(input_parse_external, init_m_state_enc=None, init_s_state_enc=None, n_iter=1):
    """
    input_parse_external is a dict i: (encoded input, encoded parse),
    where i is the iteration at which the encoded input and encoded parse
    are substtituted to the activation of respectively A_nn.MCLx and A_nn.MCLy.
    """

    init_m_state = ["idle"]
    init_s_state = ["A"]
    M_nn.MCLx.activation[
        :] = init_m_state_enc if init_m_state_enc else M_ge_alpha.ge_q(init_m_state)
    S_nn.MCLx.activation[
        :] = init_s_state_enc if init_s_state_enc else S_ge_q(init_s_state)
    A_nn.MCLx.activation[:] = 0
    A_nn.MCLy.activation[:] = 0

    activations = []
    get_acts = lambda net, layer_list: np.concatenate([getattr(net, x).activation
                                                       for x in layer_list])

    presentation_i = list(input_parse_external.keys())

    for i in range(n_iter):
        activations_t = []
        stimulus_presentation = i in presentation_i

        if stimulus_presentation:
            A_nn.MCLx.activation[:] = input_parse_external[i][0]
            A_nn.MCLy.activation[:] = input_parse_external[i][1]

        if i > 0:
            S_nn.MCLx.activate()
        S_nn.BSLbx.activate()

        tapes = np.concatenate((get_acts(S_nn, ["MCLx", "MCLy"]),
                                get_acts(A_nn, ["MCLx", "MCLy"]),))

        # print(f"{i}, 1: {tapes}")
        activations_t.append(tapes)
        activations_t.append(get_acts(S_nn, ["BSLbx"]))

        A_nn.BSLbx.activate()
        A_nn.BSLby.activate()
        A_nn.LTL.activate()
        activations_t.append(
            get_acts(A_nn, ["BSLbx", "BSLby", "LTL"]))

        B_nn.BSLbx.activate()
        B_nn.BSLby.activate()
        B_nn.LTL.activate()
        activations_t.append(
            get_acts(B_nn, ["BSLbx", "BSLby", "LTL"]))

        R_nn.BSLbx.activate()
        R_nn.BSLby.activate()
        R_nn.LTL.activate()
        activations_t.append(
            get_acts(R_nn, ["BSLbx", "BSLby", "LTL"]))

        A_nn.MCLx.activate()
        A_nn.MCLy.activate()
        tapes = np.concatenate((get_acts(S_nn, ["MCLx", "MCLy"]),
                                get_acts(A_nn, ["MCLx", "MCLy"]),))
        # print(f"{i}, 2: {tapes}")
        activations_t.append(tapes)

        if i > 0:
            M_nn.MCLx.activate()
        M_nn.BSLbx.activate()
        M_nn.BSLby.activate()
        M_nn.LTL.activate()
        # M_nn.run_net()
        activations_t.append(
            get_acts(M_nn, ["BSLbx", "BSLby", "LTL"]))
        tapes = np.concatenate((get_acts(S_nn, ["MCLx", "MCLy"]),
                                get_acts(A_nn, ["MCLx", "MCLy"]),))
        # print(f"{i}, 3: {tapes}")
        activations_t.append(tapes)

        if i > 0:
            S_nn.MCLx.activate()
        S_nn.MCLy.activate()
        S_nn.BSLbx.activate()
        S_nn.BSLby.activate()
        S_nn.LTL.activate()
        activations_t.append(
            get_acts(S_nn, ["BSLbx", "BSLby", "LTL"]))
        tapes = np.concatenate((get_acts(S_nn, ["MCLx", "MCLy"]),
                                get_acts(A_nn, ["MCLx", "MCLy"]),))
        # print(f"{i}, 4: {tapes}\n")
        activations_t.append(tapes)

        activations.append(activations_t)

    return activations

#################################################
# Run with cloud of initial conditions and plot #
#################################################

# initial conditions
init_s_state = ["A"]
init_m_state = ["idle"]
init_input_norm = ["s", "o"]
init_input_gard = ["o", "s"]
norm_inp_enc = pa_ge_i(init_input_norm)
gard_inp_enc = pa_ge_i(init_input_gard)
init_parse = ["S"]
par_enc = pa_ge_s(init_parse)
n_iter = 13
n_reps = 1000

np.random.seed(1337)

rand_m_state = np.random.uniform(
    M_ge_alpha.ge_q(init_m_state), M_ge_alpha.ge_q(init_m_state) + M_ge_alpha.ge_q.g ** (-2), n_reps)
rand_s_state = np.random.uniform(
    S_ge_q(init_s_state), S_ge_q(init_s_state) + S_ge_q.g ** (-2), n_reps)
rand_p_parse = np.random.uniform(
    pa_ge_s(init_parse), pa_ge_s(init_parse) + pa_ge_s.g ** (-2), n_reps)
rand_p_input_norm = np.random.uniform(
    pa_ge_i(init_input_norm), pa_ge_i(init_input_norm) + pa_ge_i.g ** (-3), n_reps)
rand_p_input_gard = np.random.uniform(
    pa_ge_i(init_input_gard), pa_ge_i(init_input_gard) + pa_ge_i.g ** (-3), n_reps)

get_concat = lambda acts: [np.concatenate(layers_t) for layers_t in acts]

activations_norm = [get_concat(run_whole_net({3: (r_p, r_i)}, r_m, r_s, n_iter=n_iter))
                    for r_p, r_i, r_m, r_s in zip(
                    rand_p_parse, rand_p_input_norm,
                    rand_m_state, rand_s_state)]
activations_gard = [get_concat(run_whole_net({3: (r_p, r_i)}, r_m, r_s, n_iter=n_iter))
                    for r_p, r_i, r_m, r_s in zip(
                    rand_p_parse, rand_p_input_gard,
                    rand_m_state, rand_s_state)]

activations_norm = np.array(activations_norm)
activations_gard = np.array(activations_gard)

# plt.ion()
# plot_acts(activations_norm)
# plot_acts(activations_gard)
plt.style.use("ggplot")
fig_erps = plt.figure(figsize=[8, 2])
ax_erps = fig_erps.add_subplot(111)
mean_acts_by_neuron_norm = np.mean(activations_norm, axis=2)
mean_acts_norm = np.mean(mean_acts_by_neuron_norm, axis=0)
std_acts_norm = np.std(mean_acts_by_neuron_norm, axis=0)
mean_acts_by_neuron_gard = np.mean(activations_gard, axis=2)
mean_acts_gard = np.mean(mean_acts_by_neuron_gard, axis=0)
std_acts_gard = np.std(mean_acts_by_neuron_gard, axis=0)

time_steps = np.array(range(n_iter))

ax_erps.fill_between(time_steps, mean_acts_norm - std_acts_norm,
                     mean_acts_norm + std_acts_norm, color="cornflowerblue", alpha=0.7)
ax_erps.plot(time_steps, mean_acts_norm, color="lightblue", label="Control")

ax_erps.fill_between(time_steps, mean_acts_gard - std_acts_gard,
                     mean_acts_gard + std_acts_gard, color="red", alpha=0.7)
ax_erps.plot(time_steps, mean_acts_gard,
             color="lightpink", label="Garden path")

ax_erps.set_xticks(range(n_iter))
ax_erps.set_xlim([0, n_iter - 1])
ax_erps.set_ylim([0, 0.25])
ax_erps.set_xlabel("Time step")
ax_erps.set_ylabel("Mean activation")

ax_erps.annotate(r'Stimulus' '\n presentation', xy=(2, 0.25),
                 xytext=(2, 0.28), xycoords='data',
                 annotation_clip=False, ha='center',
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3", color="black"))

ax_erps.annotate(r'Diagnosis', xy=(4, 0.10),
                 xytext=(4, 0.18), xycoords='data', color='red',
                 annotation_clip=False, ha='center',
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3", color="red"))

ax_erps.annotate(r'Repair', xy=(5, 0.15),
                 xytext=(5, 0.22), xycoords='data', color='red',
                 annotation_clip=False, ha='center',
                 arrowprops=dict(arrowstyle="simple",
                                 connectionstyle="arc3", color="red"))

legend = ax_erps.legend()
ax_erps.set_facecolor((.98, .98, .98))
legend.get_frame().set_facecolor('white')

plt.show()
# fig_erps.savefig('garden_path_cloud.pdf', bbox_inches='tight')
