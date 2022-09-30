__author__ = "Giovanni Sirio Carmantini"
__version__ = 0.1

"""

"""

import itertools as itt
from collections import Counter
from copy import copy
from typing import List, Union, Tuple

import numpy as np


def as_list(arg):
    """Convenience function used to make sure that a sequence is always
    represented by a list of symbols. Sometimes a single-symbol
    sequence is passed in the form of a string representing the
    symbol. This has to be converted to a one-item list for
    consistency, as a list of symbols is what many of the following
    functions expect.
    """
    if isinstance(arg, str):
        listed_arg = [arg]
    else:
        listed_arg = list(arg) if arg is not None else None

    return listed_arg


class FractalEncoder(object):

    """Abstract class implementing a Fractal Encoder (generalizing the
    Godel Encoders).

    """

    def __init__(self):
        raise ValueError("Not implemented")

    def encode_sequence(self, sequence):
        raise ValueError("Not implemented")

    def encode_cylinder(self, sequence):
        raise ValueError("Not implemented")

    def __call__(self, arg):
        return self.encode_sequence(arg)


class GodelEncoder(FractalEncoder):

    """Create a Godel Encoding given an alphabet. Given the encoding, you can
    then encode strings as simple numbers or cylinder sets (defined by their
    upper and lower bounds)

    :param alphabet: a list of symbols. The encoder will first associate
        one number to each symbol. In particular the symbol in alphabet[0]
        will be associated with number 0, and so on.

    """

    def __init__(self, alphabet, **kwargs):
        gamma = kwargs.get("gamma", None)
        force_power_of_two = kwargs.get("force_power_of_two", False)

        self.gamma = (
            dict([(sym, i) for i, sym in enumerate(alphabet)])
            if gamma is None
            else gamma
        )

        self.g = max(self.gamma.values()) + 1

        if force_power_of_two:
            self.g = 2 ** int(np.ceil(np.log2(self.g)))

    def encode_sequence(self, sequence):
        """Return Godel encoding of a sequence (passed as a list of strings,
        each one representing a symbol, or a string in case of a single
        symbol).
        """
        sym_list = as_list(sequence)

        if sym_list:
            return sum(
                [
                    self.gamma[sym] * pow(self.g, -i)
                    for i, sym in enumerate(sym_list, start=1)
                ]
            )

    def encode_cylinder(self, sequence, rescale=False):
        """Return Godel encoding of cylinder set of a sequence (passed as a
        list of strings, each one representing a symbol, or a string
        in case of a single symbol).


        If rescale=True the values are no
        longer bound to :math:`[0,1]`, and the most significant digit
        of the returned values in base :math:`g`, where :math:`g` is
        the number of symbols in the alphabet, is equal to the Godel
        encoding of the first symbol in the string.

        """
        sym_list = as_list(sequence)

        left_bound = 0 if not sym_list else self.encode_sequence(sym_list)
        right_bound = left_bound + pow(self.g, -len(sym_list))
        rescale_factor = [1, self.g][rescale]
        return np.array([left_bound, right_bound]) * rescale_factor


class CompactGodelEncoder(FractalEncoder):

    """Create Godel Encoder as described in our submitted paper.

    :param ge_q: A Fractal Encoder for the simulated TM states.
    :param ge_s: A Fractal Encoder for the simulated TM tape symbols.

    """

    def __init__(self, ge_q, ge_s):
        self.ge_q = ge_q
        self.ge_s = ge_s

    def encode_sequence(self, sequence):
        """Return Godel encoding of a sequence (passed as a list of strings,
        each one representing a symbol, or a string in case of a single
        symbol).
        """

        sym_list = as_list(sequence)

        if sym_list:
            return self.ge_q.encode_sequence(sym_list[0]) + self.ge_s.encode_sequence(
                sym_list[1:]
            ) * (self.ge_q.g ** -1)

    def encode_cylinder(self, sequence):
        """Return Godel encoding of cylinder set of a sequence (passed as a
        list of strings, each one representing a symbol, or a string
        in case of a single symbol).

        """

        sym_list = as_list(sequence)

        left_bound = 0 if not sym_list else self.encode_sequence(sym_list)
        right_bound = left_bound + pow(self.ge_s.g, -len(sym_list) + 1) * (
            self.g_q.g ** -1
        )
        return np.array([left_bound, right_bound])


class AbstractGeneralizedShift(object):

    """The general class."""

    def __init__(self, alpha_dod, beta_dod):
        self.alpha_dod = alpha_dod
        self.beta_dod = beta_dod

    def psi(self, alpha, beta):
        raise ValueError("Not implemented")

    def lintransf_params(self, ge_alpha, ge_beta, alpha, beta):
        raise ValueError("Not implemented")

    def iterate(self, alpha, beta, iterations=1):
        a = copy(alpha)
        b = copy(beta)
        states = [(a, b)]
        for i in range(iterations):
            a, b = self.psi(a, b)
            states.append((a, b))
        return states


class FSMGeneralizedShift(AbstractGeneralizedShift):

    """Class implementing a Generalized Shift simulating a Finite State Machine.
    For the definition of Generalized Shift see Moore (1991).

    :param states: list of states.
    :param input_symbols: list of tape symbols.
    :param delta: a dict of the form (state, symbol): new state`.
    """

    def __init__(self, states, input_symbols, delta):

        # possible symbols in DoD
        self.alpha_dod = [[x] for x in states]
        self.beta_dod = [[x] for x in input_symbols]
        self.delta = delta

    def psi(self, alpha, beta):
        """Given the inverted left part and the right part of a dotted sequence,
        apply the generalized shift on them.

        :param alpha: the inverted left part of a dotted sequence,
            as list of symbols.
        :param beta: the right part of a dotted sequence, as list of symbols.

        """

        sub_alpha, sub_beta = self.substitution(alpha, beta)
        return sub_alpha, sub_beta

    def substitution(self, alpha, beta):
        """Return new dotted sequence from substitution.

        Substitute symbols in the DoE of the dotted sequence based on the
        relevant FSM symbol substitution rule, given the symbols in the
         DoD of the Generalized Shift.
        """
        curr_state, curr_sym = alpha[0], beta[0]
        new_state, new_symbol = self.delta[(curr_state, curr_sym)], curr_sym

        new_alpha = alpha[:]
        new_beta = beta[:]

        new_alpha[0] = new_state
        new_beta[:1] = []

        return new_alpha, new_beta

    def lintransf_params(self, ge_alpha, ge_beta, alpha, beta):
        """Return two arrays containing respectively the x parameters and the
        y parameters of the linear transformation representing the GS
        action on the symbologram given a dotted sequence.

        :param ge_alpha: Fractal Encoder for left side of dotted sequence.
        :param ge_beta: Fractal Encoder for right side of dotted sequence.
        :param alpha: reversed left side of dotted sequence (as list
           of symbols).
        :param beta: right side of dotted sequence (as list of symbols).
        """
        alpha_symbols = as_list(alpha)
        beta_symbols = as_list(beta)

        ge_i, ge_q = ge_beta, ge_alpha
        g_i, g_q = ge_i.g, ge_q.g
        gamma_i, gamma_q = ge_i.gamma, ge_q.gamma
        q_old, i_old = alpha_symbols[0], beta_symbols[0]
        q_new = self.delta[(q_old, i_old)]

        lambda_x = 1.0
        lambda_y = g_i
        a_x = (-gamma_q[q_old] + gamma_q[q_new]) * (g_q ** -1)
        a_y = -gamma_i[i_old]

        return np.array([lambda_x, a_x]), np.array([lambda_y, a_y])


class PDAGeneralizedShift(AbstractGeneralizedShift):

    """Class implementing a Generalized Shift simulating a Push Down Automaton.
    For the definition of Generalized Shift see Moore (1991).

    :param stack_symbols: list of stack symbols.
    :param input_symbols: list of tape symbols.
    :param delta: a dict of the form (state, input symbol, top of stack): (new state, new top of stack)`.
    """

    def __init__(self, states, input_symbols, stack_symbols, delta):

        self.states = states
        self.input_symbols = input_symbols
        self.stack_symbols = stack_symbols

        if "e" in input_symbols or "e" in stack_symbols:
            raise ValueError(
                "'e' character used in input or stack symbols. \n"
                "The 'e' character is reserved for empty string "
                "and can't be used in input or stack symbols."
            )

        # possible symbols in DoD
        self.alpha_dod = list(itt.product(states, stack_symbols))
        self.beta_dod = [[x] for x in input_symbols]

        self.delta = delta
        if not self.check_delta_deterministic():
            raise ValueError("Delta function has to " "define a deterministic PDA.")
        self._delta_expl = self.make_delta_explicit(delta)

    def make_delta_explicit(self, delta):
        # explicit delta:
        # q, i, s -> q_new, action_on_input, action_on_stack, s_new

        input_act_f = lambda i: "do nothing" if i == "e" else "read"

        # | s\n | e          | !e           |
        # |-----+------------+--------------|
        # | e   | do nothing | push no read |
        # |-----+------------+--------------|
        # | !e  | pop        | push read    |
        # |-----+------------+--------------|
        stack_act_d = {
            (True, True): ("no read", "do nothing"),
            (True, False): ("no read" "push"),
            (False, True): ("read", "pop"),
            (False, False): ("read", "push"),
        }
        stack_act_f = lambda s, s_new: stack_act_d[(s == "e", s_new == "e")]

        delta_expl = {}
        for key, value in delta.items():
            q, i, s = key
            q_new, s_new = value
            stack_read, stack_action = stack_act_f(s, s_new)
            input_action = input_act_f(i)
            if input_action == "do nothing":
                i_syms = self.input_symbols
            else:
                i_syms = [i]

            if stack_action in ["push", "do nothing"] and stack_read == "no read":
                s_syms = self.stack_symbols
            else:
                s_syms = [s]

            for i_sym, s_sym in itt.product(i_syms, s_syms):
                delta_expl[(q, i_sym, s_sym)] = (
                    q_new,
                    input_action,
                    stack_action,
                    s_new,
                )

        return delta_expl

    def psi(self, alpha, beta):
        """Given the inverted left part and the right part of a dotted sequence,
        apply the generalized shift on them.

        :param alpha: the inverted left part of a dotted sequence,
            as list of symbols.
        :param beta: the right part of a dotted sequence, as list of symbols.

        """

        sub_alpha, sub_beta = self.substitution(alpha, beta)
        return sub_alpha, sub_beta

    def check_delta_deterministic(self):
        count_triples = Counter(
            {
                (q, i, s): 0
                for q, i, s in itt.product(
                    self.states, self.input_symbols, self.stack_symbols
                )
            }
        )
        for triple in self.delta:
            q, i, s = triple

            if (i == "e") and (s == "e"):
                count_triples.update(
                    [
                        (q, x, y)
                        for x, y in itt.product(self.input_symbols, self.stack_symbols)
                    ]
                )
            elif i == "e":
                count_triples.update([(q, x, s) for x in self.input_symbols])
            elif s == "e":
                count_triples.update([(q, i, x) for x in self.stack_symbols])
            else:
                count_triples.update([(q, i, s)])

        deterministic = True
        for x in count_triples.values():
            if x > 1:
                deterministic = False

        return deterministic

    def substitution(self, alpha, beta):
        """Return new dotted sequence from substitution.

        Substitute symbols in the DoE of the dotted sequence based on the
        relevant PDA symbol substitution rule, given the symbols in the
         DoD of the Generalized Shift.
        """

        if not beta:
            beta.append("e")
        if len(alpha) < 2:
            alpha.append("e")
        # q is state, i is input, s is stack
        q_old, i_old, s_old = alpha[0], beta[0], alpha[1]
        d_triple = q_old, i_old, s_old
        q_new, i_action, s_action, s_new = self._delta_expl[d_triple]

        new_alpha = alpha[:]
        new_beta = beta[:]

        new_alpha[0] = q_new

        if i_action == "read":
            new_beta[:1] = []

        if s_action == "push":
            new_alpha.insert(1, s_new)
        elif s_action == "pop":
            new_alpha.pop(1)
        else:  # s_action == "do nothing"
            pass

        return new_alpha, new_beta

    def lintransf_params(self, ge_alpha, ge_beta, alpha, beta):
        """Return two arrays containing respectively the x parameters and the
        y parameters of the linear transformation representing the GS
        action on the symbologram given a dotted sequence.

        :param ge_alpha: Fractal Encoder for left side of dotted sequence.
        :param ge_beta: Fractal Encoder for right side of dotted sequence.
        :param alpha: reversed left side of dotted sequence (as list
           of symbols).
        :param beta: right side of dotted sequence (as list of symbols).
        """
        alpha_syms = as_list(alpha)
        beta_syms = as_list(beta)

        q_old, i_old, s_old = alpha_syms[0], beta_syms[0], alpha_syms[1]
        d_triple = q_old, i_old, s_old
        q_new, i_action, s_action, s_new = self._delta_expl[(d_triple)]

        ge_q, ge_i, ge_s = ge_alpha.ge_q, ge_beta, ge_alpha.ge_s
        g_q, g_i, g_s = ge_q.g, ge_i.g, ge_s.g
        gamma_q, gamma_i, gamma_s = ge_q.gamma, ge_i.gamma, ge_s.gamma

        if i_action == "read":
            lambda_y = g_i
            a_y = -gamma_i[i_old]
        elif i_action == "do nothing":
            lambda_y = 1
            a_y = 0

        if s_action == "push":
            lambda_x = g_s ** -1
            a_x = (
                -gamma_q[q_old] * (g_q ** -1) * (g_s ** -1)
                + gamma_q[q_new] * (g_q ** -1)
                + gamma_s[s_new] * (g_q ** -1) * (g_s ** -1)
            )
        elif s_action == "pop":
            lambda_x = g_s
            a_x = (
                -gamma_q[q_old] * (g_q ** -1) * (g_s)
                - gamma_s[s_old] * (g_q ** -1)
                + gamma_q[q_new] * (g_q ** -1)
            )
        else:  # s_action == "do nothing"
            lambda_x = 12
            a_x = (-gamma_q[q_old] + gamma_q[q_new]) * (g_q ** -1)

        return np.array([lambda_x, a_x]), np.array([lambda_y, a_y])


class SimpleCFGeneralizedShift(AbstractGeneralizedShift):

    """A simple parser to reproduce example NDA computation in beim
    Graben, P., & Potthast, R. (2014). Universal neural field
    computation. In Neural Fields (pp. 299-318).

    :param alpha_symbols: list of symbols in the stack alphabet
    :param beta_symbols: list of symbols in the input alphabet
    :param grammar_rules: a dictionary where each entry has
        the the form "X: Y", where X is a string, and Y is
        a list of strings. Each entry corresponds to a rule
        in a Context Free Grammar, with X being a Variable
         and Y being a list of Variables and/or Terminals.

    """

    def __init__(self, alpha_symbols, beta_symbols, grammar_rules):
        AbstractGeneralizedShift.__init__(self, alpha_symbols, beta_symbols)
        self.rules = grammar_rules

    def psi(self, alpha, beta):
        """Apply Generalized Shift phi to dotted sequence."""
        alpha_symbols = as_list(alpha)
        beta_symbols = as_list(beta)
        # if can't predict (predict returns False), attach
        return self.predict(alpha_symbols, beta_symbols) or self.attach(
            alpha_symbols, beta_symbols
        )

    def predict(self, alpha, beta):
        """If a rule is present defining how to substitute the symbol in the
        alpha DoD, return the new sequence. Otherwise return False (so
        that psi can decide to apply attach instead).

        """

        new_alpha = self.rules.get(alpha[0], False) if alpha else []
        if new_alpha:
            return as_list(new_alpha) + alpha[1:] if alpha else [], beta
        else:
            return False

    def attach(self, alpha, beta):
        """If the symbols in the alpha DoD and the beta DoD are equal, pop
        them and return the new subsequences. Otherwise just return
        the original.

        """

        if (alpha and beta) and alpha[0] == beta[0]:
            return alpha[1:] if alpha else [], beta[1:] if beta else []
        else:
            return alpha, beta

    def lintransf_params(self, ge_alpha, ge_beta, alpha, beta):
        """Return two arrays containing respectively the x parameters and the
        y parameters of the linear transformation representing the GS
        action on the symbologram.

        :param ge_alpha: Fractal Encoder for left side of dotted
           sequence.

        :param ge_beta: Fractal Encoder for right side of dotted
           sequence.

        :param alpha: reversed left side of dotted sequence (as list
           of symbols or string representing one symbol).

        :param beta: right side of dotted sequence (as list of symbols
           or string representing one symbol).

        :return: [:math:`\lambda_x, a_x`], [:math:`\lambda_y, a_y`]
        :rtype: tuple(numpy.ndarray, numpy.ndarray)

        """

        alpha_head = as_list(alpha)[0] if alpha else []
        beta_head = as_list(beta)[0] if alpha else []

        enc_alpha_dod = ge_alpha.encode_sequence(alpha_head) if alpha_head else 0.0
        enc_beta_dod = ge_beta.encode_sequence(beta_head) if beta_head else 0.0

        new_alpha, new_beta = self.psi(alpha_head, beta_head)

        enc_new_alpha = ge_alpha.encode_sequence(new_alpha) if new_alpha else 0.0
        enc_new_beta = ge_beta.encode_sequence(new_beta) if new_beta else 0.0

        lambda_x = ge_alpha.g ** (-len(new_alpha) + 1)
        a_x = -enc_alpha_dod * lambda_x + enc_new_alpha
        lambda_y = ge_beta.g ** (-len(new_beta) + 1)
        a_y = -enc_beta_dod * lambda_y + enc_new_beta

        return np.array([lambda_x, a_x]), np.array([lambda_y, a_y])


class TMGeneralizedShift(AbstractGeneralizedShift):

    """Class implementing a Generalized Shift simulating a Turing Machine.
    For the definition of Generalized Shift, and the characteristics
    of GS simulating Turing Machines, see Moore (1991).

    :param states: list of states.
    :param tape_symbols: list of tape symbols.
    :param moves: a dict of the form (state, symbol): (new state, new
     symbol, movement)`, where movement is equal to either "L", "S" or
     "R", that is Left, Stay or Right.
    """

    def __init__(self, states, tape_symbols, moves):

        # possible symbols in DoD
        self.alpha_dod = list(itt.product(states, tape_symbols))
        self.beta_dod = [[x] for x in tape_symbols]

        self.moves = moves

    def psi(self, alpha, beta):
        """Given the right and the inverted left part of a dotted sequence,
        apply the generalized shift on them.

        :param alpha: the inverted left part of a dotted sequence,
            as list of symbols.
        :param beta: the right part of a dotted sequence, as list of symbols.

        """

        sub_alpha, sub_beta = self.substitution(alpha, beta)
        shift_dir = self.F(alpha, beta)
        return self.shift(shift_dir, sub_alpha, sub_beta)

    def F(self, alpha, beta):
        """Return direction of shift given simulated TM rule to apply on the
        basis of the symbols in the DoD of the dotted sequence.
        """
        shift_dir = self.moves[(alpha[0], beta[0])][2]
        return {"L": -1, "S": 0, "R": 1}[shift_dir]

    def substitution(self, alpha, beta):
        """Return new dotted sequence from substitution.

        Substitute symbols in the DoE of the dotted sequence based on the
        relevant TM symbol substitution rule, given the symbols in the
        dotted sequence DoD.
        """
        curr_state, curr_sym = alpha[0], beta[0]
        new_state, new_symbol, h_mov = self.moves[(curr_state, curr_sym)]

        new_alpha = alpha[:]
        new_beta = beta[:]

        if h_mov == "R":
            new_alpha[0:2] = [new_symbol, alpha[1]]
            new_beta[0] = new_state
        elif h_mov == "S":
            new_alpha[0] = new_state
            new_beta[0] = new_symbol
        elif h_mov == "L":
            new_alpha[0:2] = [alpha[1], new_state]
            new_beta[0] = new_symbol

        return new_alpha, new_beta

    def shift(self, shift_dir, alpha, beta):
        """Shift dotted sequence left or right (shift_dir = -1 or 1)."""

        if shift_dir == 1:
            new_alpha = [beta[0]] + alpha
            new_beta = beta[1:]

        elif shift_dir == 0:
            new_alpha = alpha
            new_beta = beta

        elif shift_dir == -1:
            new_alpha = alpha[1:]
            new_beta = alpha[0] + beta

        return new_alpha, new_beta

    def lintransf_params(
        self,
        ge_alpha: CompactGodelEncoder,
        ge_beta: GodelEncoder,
        alpha: List[str],
        beta: List[str],
    ):
        """Return two arrays containing respectively the x parameters and the
        y parameters of the linear transformation representing the GS
        action on the symbologram given a dotted sequence.

        :param ge_alpha: Fractal Encoder for left side of dotted sequence.
        :param ge_beta: Fractal Encoder for right side of dotted sequence.
        :param alpha: reversed left side of dotted sequence (as list
           of symbols).
        :param beta: right side of dotted sequence (as list of symbols).
        """
        alpha_symbols = as_list(alpha)
        beta_symbols = as_list(beta)

        move = self.moves[(alpha_symbols[0], beta_symbols[0])]

        shift_dir = self.F(alpha_symbols, beta_symbols)

        ge_s, ge_q = ge_beta, ge_alpha.ge_q
        g_n, g_q = ge_s.g, ge_q.g
        gamma_n, gamma_q = ge_s.gamma, ge_q.gamma
        q_old, s_old = alpha_symbols[0], beta_symbols[0]
        q_new, s_new = move[0], move[1]
        a_2 = alpha_symbols[1]

        if shift_dir == -1:
            lambda_x = g_n
            a_x = (
                -gamma_q[q_old] * (g_q ** -1) * (g_n)
                + gamma_q[q_new] * (g_q ** -1)
                + -gamma_n[a_2] * (g_q ** -1)
            )

            lambda_y = g_n ** -1
            a_y = (
                -gamma_n[s_old] * (g_n ** -2)
                + gamma_n[s_new] * (g_n ** -2)
                + gamma_n[a_2] * (g_n ** -1)
            )

        elif shift_dir == 0:
            lambda_x = lambda_y = 1.0
            a_x = (-gamma_q[q_old] + gamma_q[q_new]) * (g_q ** -1)
            a_y = (-gamma_n[s_old] + gamma_n[s_new]) * (g_n ** -1)

        elif shift_dir == 1:
            lambda_x = g_n ** -1
            a_x = (
                -gamma_q[q_old] * (g_q ** -1) * (g_n ** -1)
                + gamma_q[q_new] * (g_q ** -1)
                + gamma_n[s_new] * (g_n ** -1) * (g_q ** -1)
            )

            lambda_y = g_n
            a_y = -gamma_n[s_old]

        return np.array([lambda_x, a_x]), np.array([lambda_y, a_y])


class RepairGeneralizedShift(AbstractGeneralizedShift):

    """
    :param alpha_symbols: list of symbols in the stack alphabet
    :param beta_symbols: list of symbols in the input alphabet
    :param grammar_rules: a dictionary where each entry has
        the the form "X: Y", where X is a string, and Y is
        a list of strings. Each entry corresponds to a rule
        in a Context Free Grammar, with X being a Variable
         and Y being a list of Variables and/or Terminals.
    """

    def __init__(self, alpha_symbols, beta_symbols, grammar_rules):
        AbstractGeneralizedShift.__init__(self, alpha_symbols, beta_symbols)
        self.alpha_dod = [x for x in itt.product(alpha_symbols, alpha_symbols)]
        self.beta_dod = []  # beta_symbols
        self.rules = grammar_rules

    def psi(self, alpha, beta):
        """Apply Generalized Shift phi to dotted sequence."""
        alpha_symbols = as_list(alpha)
        beta_symbols = as_list(beta)

        new_alpha_doe = self.rules.get(tuple(alpha_symbols[:2]), alpha[:2])
        return as_list(new_alpha_doe) + alpha[2:], beta

    def lintransf_params(self, ge_alpha, ge_beta, alpha, beta):
        """Return two arrays containing respectively the x parameters and the
        y parameters of the linear transformation representing the GS
        action on the symbologram.

        :param ge_alpha: Fractal Encoder for left side of dotted
           sequence.

        :param ge_beta: Fractal Encoder for right side of dotted
           sequence.

        :param alpha: reversed left side of dotted sequence (as list
           of symbols or string representing one symbol).

        :param beta: right side of dotted sequence (as list of symbols
           or string representing one symbol).

        :return: [:math:`\lambda_x, a_x`], [:math:`\lambda_y, a_y`]
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """

        s_old = as_list(alpha)[:2]
        s_new, _ = self.psi(s_old, beta)

        s_old_enc = ge_alpha.encode_sequence(s_old) if s_old else 0.0
        s_new_enc = ge_alpha.encode_sequence(s_new) if s_new else 0.0

        lambda_x = 1.0
        a_x = -s_old_enc + s_new_enc
        lambda_y = 1.0
        a_y = 0.0

        return np.array([lambda_x, a_x]), np.array([lambda_y, a_y])


class NonlinearDynamicalAutomaton(object):

    """A Nonlinear Dynamical Automaton from a Generalized Shift and a Godel
    Encoding.

    :param generalized_shift: an AbstractGeneralizedShift object (as a
        base class)

    :param godel_enc_alpha: a GodelEncoder for the :math:`\\alpha
        '` reversed one-side infinite subsequence of the dotted
        sequence :math:`\\alpha . \\beta` representing a
        configuration of the Turing Machine to be simulated.

    :param godel_enc_beta: a GodelEncoder for the :math:`\\beta`
        one-side infinite subsequence of dotted sequence
        :math:`\\alpha . \\beta` representing a configuration
         of the Turing Machine to be simulated.

    """

    def __init__(
        self,
        generalized_shift: AbstractGeneralizedShift,
        godel_enc_alpha: Union[GodelEncoder, CompactGodelEncoder],
        godel_enc_beta: GodelEncoder,
        cylinder_sets: bool = False,
    ):
        self.cylinder_sets = cylinder_sets

        if not isinstance(generalized_shift, AbstractGeneralizedShift):
            raise TypeError
        else:
            self.gshift = generalized_shift

        if not (
            isinstance(godel_enc_alpha, FractalEncoder)
            and isinstance(godel_enc_beta, FractalEncoder)
        ):
            raise TypeError
        else:
            self.ga = godel_enc_alpha
            self.gb = godel_enc_beta

        self.x_leftbounds = np.array(
            [self.ga.encode_sequence(stk_s) for stk_s in self.gshift.alpha_dod]
            if self.gshift.alpha_dod
            else [0]
        )
        self.x_leftbounds.sort()
        self.y_leftbounds = np.array(
            [self.gb.encode_sequence(inp_s) for inp_s in self.gshift.beta_dod]
            if self.gshift.beta_dod
            else [0]
        )
        self.y_leftbounds.sort()

        self.flow_params_x, self.flow_params_y = self.find_flow_parameters()
        self.vflow = np.vectorize(self.flow)

    def check_cell(
            self,
            x: Union[float, Tuple[float, float]],
            y: Union[float, Tuple[float, float]],
            gencoded: bool = False
    ):
        """Return the coordinates i,j  of the input on the unit square
        partition.

        If gencoded=True, the input is assumed to be already encoded.

        """
        if gencoded:  # if the input is already encoded, do nothing
            genc_x = x
            genc_y = y
        else:  # encode it
            genc_x = self.ga.encode_sequence(x)
            genc_y = self.gb.encode_sequence(y)

        # TODO change with "left"?
        i = np.searchsorted(self.y_leftbounds, genc_y, side="right") - 1
        j = np.searchsorted(self.x_leftbounds, genc_x, side="right") - 1

        # i = i if i >= 0 else 0
        # j = j if j >= 0 else 0

        return i, j

    def find_flow_parameters(self):
        """Convert the generalized shift dynamics in dynamics on the plane,
        finding the parameters of the linear transformation for each NDA
        cell."""
        params_array_x = np.zeros((self.y_leftbounds.size, self.x_leftbounds.size, 2))
        params_array_y = np.zeros((self.y_leftbounds.size, self.x_leftbounds.size, 2))

        all_dods = itt.product(
            self.gshift.alpha_dod if self.gshift.alpha_dod else [None],
            self.gshift.beta_dod if self.gshift.beta_dod else [None],
        )

        for dod_symbols in all_dods:
            alpha_dod_symbol, beta_dod_symbol = dod_symbols
            enc_alpha = (
                self.ga.encode_sequence(alpha_dod_symbol) if alpha_dod_symbol else 0
            )
            enc_beta = (
                self.gb.encode_sequence(beta_dod_symbol) if beta_dod_symbol else 0
            )

            i, j = self.check_cell(enc_alpha, enc_beta, gencoded=True)

            params_x, params_y = self.gshift.lintransf_params(
                self.ga, self.gb, alpha_dod_symbol, beta_dod_symbol
            )

            params_array_x[i, j] = params_x
            params_array_y[i, j] = params_y

        return params_array_x, params_array_y

    def flow(self, x: float, y: float):
        """
        Given :math:`(x_t,y_t)` return
            :math:`\Psi(x_t, y_t) = (x_{t+1}, y_{t+1})`
        """

        i, j = self.check_cell(x, y, gencoded=True)
        if self.cylinder_sets:  # we apply the linear transformation defined by the left bound
            i, j = i[0], j[0]
        else:
            assert isinstance(i, int) and isinstance(j, int)

        new_x = x * self.flow_params_x[i, j, 0] + self.flow_params_x[i, j, 1]
        new_y = y * self.flow_params_y[i, j, 0] + self.flow_params_y[i, j, 1]

        return new_x, new_y

    def iterate(
            self,
            init_x: Union[float, Tuple[float, float]],
            init_y: Union[float, Tuple[float, float]],
            n_iterations: int
    ):
        """
        Apply :math:`\Psi^n(x_0, y_0)`,
            where :math:`x_0` = init_x, :math:`y_0`
            = init_y, and :math:`n` = n_iterations
        """
        x = init_x
        y = init_y
        results = [(x, y)]

        for _ in range(n_iterations):
            x, y = self.flow(x, y)
            results.append((x, y))

        return results
