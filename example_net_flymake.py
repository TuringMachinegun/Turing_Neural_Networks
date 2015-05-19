__author__ = 'Giovanni Carmantini'

import NDA
import NeuralTM as nTM


stack_symbols = ["NP", "V", "VP", "S"]
input_symbols = ["NP", "V"]
rules = (["S -> NP VP", "VP -> V NP"])

#  generalized shift from context free grammar
gshift = NDA.SimpleCFGeneralizedShift(stack_symbols, input_symbols, rules)

#  Godel encoding for stack and input
ga = NDA.GodelEncoder(stack_symbols)
gb = NDA.GodelEncoder(input_symbols)

print ga.get_lintransf_params(["NP"], ["NP", "V"])
#  NDS from generalized shift and Godel encoding
nda = NDA.NonlinearDynamicalAutomaton(generalized_shift=gshift,
                                      godel_enc_alpha=ga,
                                      godel_enc_beta=gb)

#  Neural Turing Machine from NDA
example_net = nTM.NeuralTM(nda.flow_params_x, nda.flow_params_y)
acts = example_net.run_net(init_x=[0.75, 1], init_y=[0.25, 0.375],
                           n_iterations=5)

#  print the activations of neurons in the input layer for each iteration
for x, y in acts:
    print x, y
