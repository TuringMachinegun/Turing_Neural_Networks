__author__ = "Jone Uria Albizuri"
"""
In this file, we create two neural networks, both encoding the simple parser from [1] via the process described in [2].
The encoding of the two networks differs for the chosen ordering of symbols in the Gödel encodings of input and
stack.

We then compute the Amari's observable of the two networks running on the same symbolic input, and plot it to
 show it is not invariant to Gödel recoding.

Finally we compute the step function observable for the same networks and inputs, showing its invariance under
 Gödel recoding.

[1] beim Graben, P., & Potthast, R. (2014). Universal neural field computation. In Neural Fields (pp. 299-318).
[2] Carmantini, G.S. et al. (2017). A modular architecture for transparent computation in recurrent neural networks.
    In Neural Networks, 85 (pp.85-105).
"""

import matplotlib.pyplot as plt
import numpy as np

from tnnpy import NeuralTM
from tnnpy.step_function_observable import step_function_blank, apply_step_function
from tnnpy.symdyn import GodelEncoder, SimpleCFGeneralizedShift, NonlinearDynamicalAutomaton

PARSER_DESCRIPTION = {
    "S": ["NP", "VP"],
    "VP": ["V", "NP"]
}
INIT_STACK = ["S"]
INIT_INPUT = ["NP", "V", "NP"]


# 1. Encode two machines differing only for their Gödel encoding (i.e. different order of input and stack symbols)

def encode_machine(input_symbols, stack_symbols, parser_descr):
    """Given input and stack symbols, and a parser description, return the stack and input Gödel encodings,
    as well as a neural network implementing the parser as described"""
    # Godel Encoders
    ge_s = GodelEncoder(stack_symbols, force_power_of_two=True)
    ge_i = GodelEncoder(input_symbols, force_power_of_two=True)

    # CFG -> GS
    cfg_gs = SimpleCFGeneralizedShift(stack_symbols, input_symbols, parser_descr)
    # GS -> NDA
    nda = NonlinearDynamicalAutomaton(cfg_gs, ge_s, ge_i, cylinder_sets=True)

    # NDA -> R-ANN
    cfg_nn = NeuralTM(nda, cylinder_sets=True)
    return ge_s, ge_i, cfg_nn


ge_s_m1, ge_i_m1, cfg_nn_m1 = encode_machine(
    input_symbols=[" ", "NP", "V"],
    stack_symbols=[" ", "NP", "V", "VP", "S"],
    parser_descr=PARSER_DESCRIPTION
)

ge_s_m2, ge_i_m2, cfg_nn_m2 = encode_machine(
    input_symbols=[" ", "V", "NP"],
    stack_symbols=[" ", "VP", "S", "V", "NP"],
    parser_descr=PARSER_DESCRIPTION
)


# 2. Compute Amari's observable for the two machines

def compute_amari_observable(n_iter, nnet, encoded_init_stack, encoded_init_input):
    mean_acts = np.zeros(n_iter)
    nnet.set_init_cond(encoded_init_stack, encoded_init_input)
    for j in range(n_iter):
        MCL_acts = np.concatenate((nnet.MCLx.activation, nnet.MCLy.activation))
        nnet.run_net()
        all_acts = np.concatenate(
            (
                MCL_acts,
                nnet.BSLbx.activation,
                nnet.BSLby.activation,
                nnet.LTL.activation,
            )
        )
        mean_acts[j] = np.mean(all_acts)
    return mean_acts


n_iter = 6
amari_m1 = compute_amari_observable(
    n_iter=n_iter,
    nnet=cfg_nn_m1,
    encoded_init_stack=ge_s_m1.encode_cylinder(INIT_STACK),
    encoded_init_input=ge_i_m1.encode_cylinder(INIT_INPUT)
)
print(amari_m1)

amari_m2 = compute_amari_observable(
    n_iter=n_iter,
    nnet=cfg_nn_m2,
    encoded_init_stack=ge_s_m2.encode_cylinder(INIT_STACK),
    encoded_init_input=ge_i_m2.encode_cylinder(INIT_INPUT)
)
print(amari_m2)


# 3. compute step function observable for the two machines

def compute_step_function_observable(
        n_iter: int, nnet: NeuralTM, encoded_init_stack, encoded_init_input, partition_rect, C):
    cfg_states = nnet.run_net(init_x=encoded_init_stack, init_y=encoded_init_input, n_iterations=n_iter)

    step_vals = []
    last_state = None
    for i, cfg_state in enumerate(cfg_states):
        if np.array_equal(cfg_state, last_state):
            break
        x, y = cfg_state[0][0], cfg_state[1][0]
        step_vals.append(
            apply_step_function([x, y], partition_rect, C)
        )
        last_state = cfg_state
    return step_vals


partition_rect, C = step_function_blank(
    x_symbols=[str(i) for i, _ in enumerate(ge_s_m1.gamma)],
    y_symbols=[str(i) for i, _ in enumerate(ge_i_m1.gamma)],
    DoD=[2, 3]
)
step_vals_m1 = compute_step_function_observable(
    n_iter=n_iter,
    nnet=cfg_nn_m1,
    encoded_init_stack=ge_s_m1.encode_cylinder(INIT_STACK),
    encoded_init_input=ge_i_m1.encode_cylinder(INIT_INPUT),
    partition_rect=partition_rect,
    C=C
)

step_vals_m2 = compute_step_function_observable(
    n_iter=n_iter,
    nnet=cfg_nn_m2,
    encoded_init_stack=ge_s_m2.encode_cylinder(INIT_STACK),
    encoded_init_input=ge_i_m2.encode_cylinder(INIT_INPUT),
    partition_rect=partition_rect,
    C=C
)

# 3. Plot results

base_z_order = 10
plt.style.use("ggplot")
fig_amari, axs = plt.subplots(2, figsize=[5, 5])

t = np.linspace(0, 1, len(amari_m1))
axs[0].set_xticks(t, labels=['1', '2', '3', '4', '5', '6'])
axs[1].set_xticks(t, labels=['1', '2', '3', '4', '5', '6'])
axs[1].set_xlabel("iteration")
axs[1].set_ylabel("Amari's observable")
axs[0].set_ylabel("Amari's observable")
axs[0].plot(t, amari_m1, label="Gödelization by $\gamma$")
axs[1].plot(t, amari_m2, color="blue", label="Gödelization by $\delta$")
axs[0].legend()
axs[1].legend()

plt.savefig('amaris_observable_under_recoding.eps')

fig_step, axs = plt.subplots(2, figsize=[5, 5])
t = np.linspace(0, 1, len(step_vals_m1))
axs[0].set_xticks(t, labels=['1', '2', '3', '4', '5', '6'])
axs[1].set_xticks(t, labels=['1', '2', '3', '4', '5', '6'])
axs[1].set_xlabel("iteration")
axs[1].set_ylabel("step function")
axs[0].set_ylabel("step function")
print("this is C:", C)
print(len(C))
axs[0].plot(t, step_vals_m1, label="Gödelization $\gamma$")
axs[1].plot(t, step_vals_m2, color="blue", label="Gödelization $\delta$")
axs[0].legend()
axs[1].legend()
plt.savefig('step_fn_observable_under_recoding.eps')
plt.show()
