import NDA
import NeuralTM as nTM

tape_symbols = ["b", "0", "1", "n"]
states = ["halt", "q_0", "q_1", "q_2"]
tm_descr = {("q_0", "b"): ("q_1", "b", "L"), ("q_0", "0"): ("q_0", "0", "R"),
            ("q_0", "1"): ("q_0", "1", "R"), ("q_0", "n"): ("q_0", "n", "R"),
            ("q_1", "b"): ("q_2", "1", "R"), ("q_1", "0"): ("q_2", "1", "L"),
            ("q_1", "1"): ("q_1", "0", "L"), ("q_1", "n"): ("q_0", "n", "R"),
            ("q_2", "b"): ("halt", "b", "L"), ("q_2", "0"): ("q_2", "0", "R"),
            ("q_2", "1"): ("q_2", "1", "R"), ("q_2", "n"): ("q_0", "n", "R"),
            ("halt", "b"): ("halt", "b", "L"), ("halt", "0"): ("halt", "0", "R"),
            ("halt", "1"): ("halt", "1", "R"), ("halt", "n"): ("q_0", "n", "R")}

ge_q = NDA.GodelEncoder(states)
ge_n = NDA.GodelEncoder(tape_symbols)

ge_alpha = NDA.alphaGodelEncoder(ge_q, ge_n)
ge_beta = ge_n

init_state = ge_alpha.encode_string(["q_0", "b"])
init_tape = ge_n.encode_string(list("0111"))

tm_gs = NDA.TMGeneralizedShift(states, tape_symbols, tm_descr)
nda = NDA.NonlinearDynamicalAutomaton(tm_gs, ge_alpha, ge_beta)

results = nda.iterate(init_state, init_tape, 10)

#  Neural Turing Machine from NDA
example_net = nTM.NeuralTM(nda.flow_params_x, nda.flow_params_y)
acts = example_net.run_net(init_x=[init_state, init_state], init_y=[init_tape, init_tape],
                           n_iterations=5)

#  print the activations of neurons in the input layer for each iteration
for x, y in acts:
    print x, y


def tape_decoder(tape_symbols, encoded_tape, g_number):
    decoded_syms = []

    tape = encoded_tape

    for i in range(15):
        tape *= g_number
        msd = int(tape)
        decoded_syms.append(tape_symbols[msd])
        tape -= msd

    return decoded_syms
