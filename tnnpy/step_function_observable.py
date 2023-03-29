__author__ = "Jone Uria Albizuri"

from itertools import permutations, product
from typing import List, Tuple

import numpy as np

from tnnpy import GodelEncoder


def decimal_to_base(x, m, max_length=10, power_of_two=False):
    """Given x a number on the interval [0,1], m the size of the alphabet it returns the sequence
    of length max_length on the alphabet [0,1,...,m-1] that Gödelized by the identity gives x."""
    n = max_length
    if power_of_two == True:
        m = 2 ** int(np.ceil(np.log2(m)))

    s = x * m ** n
    M = np.zeros(n)
    for i in range(len(M)):
        M[i] = int(s % m)
        s = s // m
    A = [int(a) for a in M]
    B = list(map(str, A))
    B.reverse()
    return B


def get_invariant_partition_1d_blank(alphabet: List[str], sequence: List[str]) -> List[np.ndarray]:
    """Return Godel-encoded cylinders for sequence, applying all possible enumerations for alphabet symbols in encoding.

    :param alphabet: A list of symbols comprising an alphabet
    :param sequence: A list of symbols from the alphabet
    :return: encoded cylinders for sequence when applying Godel encodings with all possible enumerations, with each
        element of returned list being of the form np.ndarray([seq_left_bound, seq_right_bound])
    """
    ## Same code we had but just permuting the alphabet symbols starting from the second letter
    cylinders = []
    g = 2 ** int(np.ceil(np.log2(len(alphabet))))
    new_alphabet = alphabet[1:len(alphabet)]
    for alpha_p in permutations(new_alphabet, len(new_alphabet)):
        alpha_p1 = [alphabet[0]]
        alpha_p2 = list(range(len(alphabet), g))
        alpha_p3 = alpha_p1 + list(alpha_p) + alpha_p2
        cylinders.append(GodelEncoder(alpha_p3).encode_cylinder(sequence))
    return list(sorted(cylinders, key=lambda x: x[0]))


def get_invariant_partition_2d_blank(
        alpha_alphabet: List[str],
        beta_alphabet: List[str],
        alpha_sequence: List[str],
        beta_sequence: List[str]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return Godel-encoded cylinder sets for dotted sequence, applying all possible enumerations for symbols
    in alpha and beta alphabets.

    :param alpha_alphabet: A list of symbols comprising the alphabet for the left part of the dotted sequence
    :param beta_alphabet: A list of symbols comprising the alphabet for the right part of the dotted sequence
    :param alpha_sequence: left part of the dotted sequence (right-to-left from the dot)
    :param beta_sequence: right part of the dotted sequence
    :return: encoded cylinder sets for dotted sequence when applying Godel encodings with all possible enumerations,
        with each element in the returned list being of the form:
        `np.ndarray([alpha_seq_left_bound, alpha_seq_right_bound]),
         np.ndarray([beta_seq_left_bound, beta_seq_right_bound])`
    """
    alpha_cyl = get_invariant_partition_1d_blank(alpha_alphabet, alpha_sequence)
    beta_cyl = get_invariant_partition_1d_blank(beta_alphabet, beta_sequence)
    partition = list(product(alpha_cyl, beta_cyl))
    return partition


def partition_rectangle_blank(x_symbols, y_symbols, DoD):
    """Given the alphabet on the right hand said of the dot (x_symbols) and the one on the
    left hand side of the dot (y_symbols) plus the lengths of the Domain of Dependance [r,s]
    it returns an array with all the different partitions according to the patterns of equality
    It assumes that the first symbol will always be blank and just permutes and computes the pattern of
    equality for the rest.
    """
    I = []
    m_x = len(x_symbols)
    m_y = len(y_symbols)
    alpha_symbols = list(map(str, range(m_x)))
    beta_symbols = list(map(str, range(m_y)))
    x_bounds = product(alpha_symbols, repeat=DoD[0])
    for alpha_seq in x_bounds:
        y_bounds = product(beta_symbols, repeat=DoD[1])
        for beta_seq in y_bounds:
            k = 0
            equality_pattern = get_invariant_partition_2d_blank(alpha_symbols, beta_symbols, alpha_seq, beta_seq)
            for i in range(len(I)):
                if not np.array_equal(equality_pattern, I[i]):
                    k = k + 1
            if k == (len(I)):
                I.append(equality_pattern)
    return I


def step_function_blank(x_symbols, y_symbols, DoD):
    """Given the symbols in both alphabets and the DoD computes the invariant partition
    and assigns a random number corresponding to each part"""
    PR = partition_rectangle_blank(x_symbols, y_symbols, DoD)
    C = np.random.permutation(len(PR))  # range(len(PR))
    return [PR, C]


def apply_step_function(state, PR, C):
    """Given a state (x,y), the paritition rectangle and the assigment for each part, it returns the number
    corresponding to the equality pattern of that state"""

    for j in range(len(PR)):
        for k in range(len(PR[j])):
            if ((PR[j][k][0][0] <= state[0]) and (state[0] < PR[j][k][0][1]) and (PR[j][k][1][0] <= state[1]) and (
                    state[1] < PR[j][k][1][1])):
                m = j
    return C[m] / len(PR)
