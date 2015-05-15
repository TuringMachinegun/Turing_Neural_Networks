__author__ = 'Giovanni Carmantini'

"""
In this library I implement the classes needed to make an NDA, in particular:
a Godel Encoder, applying a Godel Encoding to an alphabet
a Generalized Shift for simple parsers like the one used by Peter in his example
an NDA from the Generalized Shift and the Godel Encoder
"""

import numpy as np
import itertools as itt


class GodelEncoder(object):
    """ Create a Godel Encoding given an alphabet.
    Given the encoding, you can then encode strings as simple numbers
    or cylinder sets (defined by their upper and lower bounds)

    :param alphabet: a list of symbols. The encoder will first associate one number to each symbol.
    In particular the symbol in alphabet[0] will be associated with number 0, and so on.
    """

    def __init__(self, alphabet):
        self.symbols_dict = dict([(sym, i) for i, sym in enumerate(alphabet)])
        self.g = len(alphabet) if hasattr(alphabet, "__len__") else alphabet.size  # Godel number for encoding

    def encode_string(self, symbols_list):
        """
        Return Godel encoding of a string (passed as a list of symbols or a string)
        """
        sym_list = symbols_list[:]
        sym_list = [sym_list] if isinstance(sym_list, str) else sym_list
        return sum([self.symbols_dict[sym] * pow(self.g, -i) for i, sym in enumerate(sym_list, start=1)])

    def encode_cylinder(self, symbols_list, rescale=False):
        """Return Godel encoding of cylinder set of a string (passed as a list
        of symbols or a string). If rescale=True the values are no
        longer bound to :math:`[0,1]`, and the most significant digit
        of the returned values in base :math:`g`, where :math:`g` is
        the number of symbols in the alphabet, is equal to the Godel
        encoding of the first symbol in the string.

        """
        sym_list = symbols_list[:]
        sym_list = [sym_list] if isinstance(sym_list, str) else sym_list
        left_bound = self.encode_string(sym_list)
        right_bound = left_bound + pow(self.g, -len(sym_list))
        rescale_factor = [1, self.g][rescale]
        return np.array([left_bound, right_bound])*rescale_factor


class TMGeneralizedShift(object):
    """
    The general class, not really useful now. It will be when I'll start working with TMs.
    """
    def __init__(self, alpha_symbols, beta_symbols):
        self.alpha_symbols = alpha_symbols
        self.beta_symbols = beta_symbols

    def psi(self, alpha, beta):
        raise ValueError("Not implemented")


class SimpleCFGeneralizedShift(TMGeneralizedShift):
    """
    A simple parser to reproduce Peter's example.

    :param alpha_symbols: list of symbols in the stack alphabet
    :param beta_symbols: list of symbols in the input alphabet
    """

    def __init__(self, alpha_symbols, beta_symbols, grammar_rules):
        TMGeneralizedShift.__init__(self, alpha_symbols, beta_symbols)
        self.rules = self.cfg_rules_to_dict(grammar_rules)

    def psi(self, s, i):
        """Applies Generalized Shift function to DoD (so only two symbols). I
        could make it general and appliable to the whole sequence, but
        for the moment I have no reason to implement this
        complication.
        """
        return self.predict(s, i) or self.attach(s, i)  # if can't predict (predict returns False), attach

    def predict(self, alpha_symbol, beta_symbol):
        """
        If a rule is present defining how to substitute alpha_symbol,
        return the substitution and the beta symbol.
        Otherwise return False.
        """
        new_alpha = self.rules.get("".join(alpha_symbol), False)
        if new_alpha:
            return new_alpha, beta_symbol
        else:
            return False

    def attach(self, alpha_symbol, beta_symbol):
        """
        Return the equivalent of two empty strings in alpha and beta,
        if alpha_symbol and beta_symbol are the same symbol.
        Otherwise return the two original symbols.
        """
        return[(alpha_symbol, beta_symbol), ([], [])][alpha_symbol == beta_symbol]

    @staticmethod
    def cfg_rules_to_dict(rules):
        """Return a dictionary where each entry associates a Variable to a
        string of Terminals.

        :param rules: a list of strings in the form "X -> Y" where X is a Variable, Y is a string of Variables and/or Terminals.

        """
        producer_produced = [rule.split("->") for rule in rules]
        rules_dict = dict([(x.strip(), y.split()) for x, y in producer_produced])
        return rules_dict


class NonlinearDynamicalAutomaton(object):
    """
    A Nonlinear Dynamical Automaton from a Generalized Shift and a Godel Encoding
    
    :param generalized_shift: a TMGeneralizedShift object, i.e. a Generalized 
	Shift simulating a Turing Machine 
    :param godel_enc_alpha: a GodelEncoder for the :math:`\\alpha '` reversed one-side infinite subsequence
        of the dotted sequence :math:`\\alpha . \\beta` representing a configuration of the Turing Machine to be simulated.
    :param godel_enc_beta: a GodelEncoder for the :math:`\\beta` one-side infinite subsequence of
        the dotted sequence :math:`\\alpha . \\beta` representing a configuration of the Turing Machine to be simulated.

    """

    def __init__(self, generalized_shift, godel_enc_alpha, godel_enc_beta):

        if not isinstance(generalized_shift, TMGeneralizedShift):
            raise TypeError
        else:
            self.gshift = generalized_shift

        if not (isinstance(godel_enc_alpha, GodelEncoder)
                and isinstance(godel_enc_beta, GodelEncoder)):
            raise TypeError
        else:
            self.ga = godel_enc_alpha
            self.gb = godel_enc_beta

        self.apha_lb_partns = np.array([self.ga.encode_string(stk_s) for stk_s in self.gshift.alpha_symbols])
        self.beta_lb_partns = np.array([self.gb.encode_string(inp_s) for inp_s in self.gshift.beta_symbols])

        self.flow_params_x, self.flow_params_y = self.calculate_flow_parameters()
        self.vflow = np.vectorize(self.flow)

    def check_cell(self, x, y, gencoded=False):
        """
        Return the coordinates i,j  of the input on the unit square partition.
        If gencoded=True, the input is assumed to be already encoded.
        """
        if gencoded:  # if the input is already encoded, do nothing, else encode it
            genc_x = x
            genc_y = y
        else:
            genc_x = self.ga.encode_string(x)
            genc_y = self.gb.encode_string(y)

        alpha_cell = np.searchsorted(self.apha_lb_partns, genc_x, side="right") - 1
        beta_cell = np.searchsorted(self.beta_lb_partns, genc_y, side="right") - 1

        return alpha_cell, beta_cell

    def calculate_flow_parameters(self):
        """
        Converts the generalized shift dynamics in dynamics on the plane,
        finding the parameters of the linear transformation for each NDA cell
        """
        params_array_x = np.zeros((self.apha_lb_partns.size, self.beta_lb_partns.size, 2))
        params_array_y = np.zeros((self.apha_lb_partns.size, self.beta_lb_partns.size, 2))

        for DoD in itt.product(self.gshift.alpha_symbols, self.gshift.beta_symbols):

            alpha_dod, beta_dod = DoD
            alpha_doe, beta_doe = self.gshift.psi(alpha_dod, beta_dod)
            enc_alpha_dod, enc_alpha_doe = self.ga.encode_cylinder(alpha_dod), self.ga.encode_cylinder(alpha_doe)
            enc_beta_dod, enc_beta_doe = self.gb.encode_cylinder(beta_dod), self.gb.encode_cylinder(beta_doe)

            p = self.check_cell(enc_alpha_dod[0], enc_beta_dod[0], gencoded=True)

            params_array_x[p[0], p[1]] = np.array([np.linalg.solve(zip(enc_alpha_dod, [1, 1]), enc_alpha_doe)])
            params_array_y[p[0], p[1]] = np.array([np.linalg.solve(zip(enc_beta_dod, [1, 1]), enc_beta_doe)])

        return params_array_x, params_array_y

    def flow(self, x, y):
        """
        Given :math:`(x_t,y_t)` returns :math:`\Psi(x_t, y_t) = (x_{t+1}, y_{t+1})`
        """

        p_x, p_y = self.check_cell(x, y, gencoded=True)

        new_x = x*self.flow_params_x[p_x, p_y, 0] + self.flow_params_x[p_x, p_y, 1]
        new_y = y*self.flow_params_y[p_x, p_y, 0] + self.flow_params_y[p_x, p_y, 1]

        return new_x, new_y

    def iterate(self, init_x, init_y, n_iterations):
        """
        Applies :math:`\Psi^n(x_0, y_0)`, where :math:`x_0` = init_x, :math:`y_0` = init_y, and :math:`n` = n_iterations
        """
        x = init_x
        y = init_y
        results = [(x, y)]

        for _ in range(n_iterations):
            x, y = self.flow(x, y)
            results.append((x, y))

        return results
