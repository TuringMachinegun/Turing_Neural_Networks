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
        self.gamma = dict([(sym, i) for i, sym in enumerate(alphabet)])

        if hasattr(alphabet, "__len__"):
            self.g = len(alphabet)
        else:
            self.g = alphabet.size

    def encode_string(self, symbols_list):
        """
        Return Godel encoding of a string (passed as a list of symbols or a string)
        """
        sym_list = symbols_list[:]
        sym_list = [sym_list] if isinstance(sym_list, str) else sym_list

        if sym_list:
            return sum([self.gamma[sym] * pow(self.g, -i) for i, sym in enumerate(sym_list, start=1)])

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
        left_bound = 0 if not sym_list else self.encode_string(sym_list)
        right_bound = left_bound + pow(self.g, -len(sym_list))
        rescale_factor = [1, self.g][rescale]
        return np.array([left_bound, right_bound])*rescale_factor


class alphaGodelEncoder(GodelEncoder):

    def __init__(self, ge_q, ge_n):
        self.ge_q = ge_q
        self.ge_n = ge_n

    def encode_string(self, symbols_list):
        sym_list = symbols_list[:]
        sym_list = [sym_list] if isinstance(sym_list, str) else sym_list

        if sym_list:
            return self.ge_q.encode_string(symbols_list[0]) + \
                self.ge_n.encode_string(symbols_list[1:])*(self.ge_q.g**-1)

    def encode_cylinder(self, symbols_list):
        """Return Godel encoding of cylinder set of a string (passed
        as a list of symbols or a string).
        """
        sym_list = symbols_list[:]
        sym_list = [sym_list] if isinstance(sym_list, str) else sym_list
        left_bound = 0 if not sym_list else self.encode_string(sym_list)
        right_bound = left_bound + pow(self.ge_n.g, -len(sym_list)+1)*(self.g_q.g**-1)
        return np.array([left_bound, right_bound])


class AbstractGeneralizedShift(object):
    """
    The general class.
    """
    def __init__(self, alpha_dod, beta_dod):
        self.alpha_dod = alpha_dod
        self.beta_dod = beta_dod

    def psi(self, alpha, beta):
        raise ValueError("Not implemented")

    def lintransf_params(self, ge_alpha, ge_beta, alpha_dod, beta_dod):
        raise ValueError("Not implemented")


class TMGeneralizedShift(AbstractGeneralizedShift):
    """Class implementing a Generalized Shift simulating a Turing Machine.

    :param alpha_symbols: list of symbols that can occur in alpha
        (states + tape symbols)
    :param beta_symbols: list of symbols that can occur in beta (tape symbols)
    :moves: a dict of the form :math:`(q_1, \sigma_1): (q_2, \sigma_2, M)`
        where M is a movement of the read-write head
    """
    def __init__(self, states, tape_symbols, moves):
        self.alpha_dod = list(itt.product(states, tape_symbols))
        self.beta_dod = tape_symbols
        self.moves = moves

    def psi(self, alpha, beta):
        """Given the right and the inverted left part of a dotted sequence,
        apply the generalized shift on them.

        :param alpha: the inverted left part of a dotted sequence, a list of symbols.
        :param beta: the right part of a dotted sequence, a list of symbols.
        """

        sub_alpha, sub_beta = self.substitution(alpha, beta)
        shift_dir = self.F(alpha, beta)
        return self.shift(shift_dir, sub_alpha, sub_beta)

    def F(self, alpha, beta):
        shift_dir = self.moves[(alpha[0], beta[0])][2]
        if shift_dir == "R":
            return 1
        if shift_dir == "L":
            return -1

    def substitution(self, alpha, beta):
        curr_state, curr_sym = alpha[0], beta[0]
        new_state, new_symbol, h_mov = self.moves[(curr_state, curr_sym)]

        new_alpha = alpha[:]
        new_beta = beta[:]

        if h_mov == "R":
            new_alpha[0:2] = [new_symbol, alpha[1]]
            new_beta[0] = new_state
        elif h_mov == "L":
            new_alpha[0:2] = [alpha[1], new_state]
            new_beta[0] = new_symbol

        return new_alpha, new_beta

    def shift(self, shift_dir, alpha, beta):

        if shift_dir == 1:
            new_alpha = [beta[0]] + alpha
            new_beta = beta[1:]
        elif shift_dir == -1:
            new_alpha = alpha[1:]
            new_beta = alpha[0] + beta

        return new_alpha, new_beta

    def lintransf_params(self, ge_alpha, ge_beta, alpha_dod, beta_dod):

        move = self.moves[(alpha_dod[0], beta_dod)]

        shift_dir = self.F(alpha_dod, beta_dod)

        ge_n, ge_q = ge_beta, ge_alpha.ge_q
        g_n, g_q = ge_n.g, ge_q.g
        gamma_n, gamma_q = ge_n.gamma, ge_q.gamma
        q_old, s_old = alpha_dod[0], beta_dod[0]
        q_new, s_new = move[0], move[1]
        a_2 = alpha_dod[1]

        if shift_dir == -1:
            lambda_x = g_n
            a_x = -gamma_q[q_old]*(g_q**-1)*(g_n) + \
                gamma_q[q_new]*(g_q**-1) + \
                -gamma_n[a_2]*(g_q**-1)

            lambda_y = g_n**-1
            a_y = -gamma_n[s_old]*(g_n**-2) + \
                gamma_n[s_new]*(g_n**-2) + \
                gamma_n[a_2]*(g_n**-1)

        elif shift_dir == 1:
            lambda_x = g_n**-1
            a_x = -gamma_q[q_old]*(g_q**-1)*(g_n**-1) + \
                gamma_q[q_new]*(g_q**-1) + \
                gamma_n[s_new]*(g_n**-1)*(g_q**-1)

            lambda_y = g_n
            a_y = - gamma_n[s_old]

        return np.array([lambda_x, a_x]), np.array([lambda_y, a_y])


class SimpleCFGeneralizedShift(AbstractGeneralizedShift):

    """
    A simple parser to reproduce Peter's example.

    :param alpha_symbols: list of symbols in the stack alphabet
    :param beta_symbols: list of symbols in the input alphabet
    :param grammar_rules: a list of strings in the form "X -> Y" where X is a Variable,
        Y is a string of Variables and/or Terminals.
    """

    def __init__(self, alpha_dod, beta_dod, grammar_rules):
        AbstractGeneralizedShift.__init__(self, alpha_dod, beta_dod)
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

    def lintransf_params(self, ge_alpha, ge_beta, alpha_dod, beta_dod):
        """ TODO: make this function"""
        pass

    @staticmethod
    def cfg_rules_to_dict(rules):
        """Return a dictionary where each entry associates a Variable to a
        string of Terminals.

        :param rules: a list of strings in the form "X -> Y" where X
           is a Variable, Y is a string of Variables and/or Terminals.

        """
        producer_produced = [rule.split("->") for rule in rules]
        rules_dict = dict([(x.strip(), y.split()) for x, y in producer_produced])
        return rules_dict


class NonlinearDynamicalAutomaton(object):
    """
    A Nonlinear Dynamical Automaton from a Generalized Shift and a Godel Encoding

    :param generalized_shift: a AbstractGeneralizedShift object, i.e. a Generalized
        Shift simulating a Turing Machine
    :param godel_enc_alpha: a GodelEncoder for the :math:`\\alpha '` reversed
        one-side infinite subsequence of the dotted sequence
        :math:`\\alpha . \\beta` representing a configuration
         of the Turing Machine to be simulated.
    :param godel_enc_beta: a GodelEncoder for the :math:`\\beta`
        one-side infinite subsequence of dotted sequence
        :math:`\\alpha . \\beta` representing a configuration
         of the Turing Machine to be simulated.
    """

    def __init__(self, generalized_shift, godel_enc_alpha, godel_enc_beta):

        if not isinstance(generalized_shift, AbstractGeneralizedShift):
            raise TypeError
        else:
            self.gshift = generalized_shift

        if not (isinstance(godel_enc_alpha, GodelEncoder)
                and isinstance(godel_enc_beta, GodelEncoder)):
            raise TypeError
        else:
            self.ga = godel_enc_alpha
            self.gb = godel_enc_beta

        self.x_leftbounds = np.array([self.ga.encode_string(stk_s) for
                                      stk_s in self.gshift.alpha_dod])
        self.x_leftbounds.sort()
        self.y_leftbounds = np.array([self.gb.encode_string(inp_s) for
                                      inp_s in self.gshift.beta_dod])
        self.y_leftbounds.sort()

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
            
        # change with "left"?
        alpha_cell = np.searchsorted(self.x_leftbounds, genc_x, side="right") - 1
        beta_cell = np.searchsorted(self.y_leftbounds, genc_y, side="right") - 1

        return alpha_cell, beta_cell

    def calculate_flow_parameters(self):
        """
        Converts the generalized shift dynamics in dynamics on the plane,
        finding the parameters of the linear transformation for each NDA cell
        """
        params_array_x = np.zeros((self.x_leftbounds.size,
                                   self.y_leftbounds.size, 2))
        params_array_y = np.zeros((self.x_leftbounds.size,
                                   self.y_leftbounds.size, 2))

        for DoD in itt.product(self.gshift.alpha_dod, self.gshift.beta_dod):
            alpha, beta = DoD
            enc_alpha = self.ga.encode_string(alpha)
            enc_beta = self.gb.encode_string(beta)

            p = self.check_cell(enc_alpha, enc_beta, gencoded=True)

            params_x, params_y = self.gshift.lintransf_params(self.ga, self.gb, 
                                                              alpha, beta)

            params_array_x[p[0], p[1]] = params_x
            params_array_y[p[0], p[1]] = params_y

        return params_array_x, params_array_y

    def flow(self, x, y):
        """
        Given :math:`(x_t,y_t)` returns :math:`\Psi(x_t, y_t) = (x_{t+1}, y_{t+1})`
        """

        p_x, p_y = self.check_cell(x, y, gencoded=True)

        new_x = (x*self.flow_params_x[p_x, p_y, 0] +
                 self.flow_params_x[p_x, p_y, 1])
        new_y = (y*self.flow_params_y[p_x, p_y, 0] +
                 self.flow_params_y[p_x, p_y, 1])

        return new_x, new_y

    def iterate(self, init_x, init_y, n_iterations):
        """
        Applies :math:`\Psi^n(x_0, y_0)`, where :math:`x_0` = init_x, :math:`y_0`
            = init_y, and :math:`n` = n_iterations
        """
        x = init_x
        y = init_y
        results = [(x, y)]

        for _ in range(n_iterations):
            x, y = self.flow(x, y)
            results.append((x, y))

        return results
