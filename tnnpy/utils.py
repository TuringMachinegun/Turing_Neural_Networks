from itertools import permutations, product
from typing import List, Tuple

import numpy as np

from tnnpy import GodelEncoder


def get_invariant_partition_1d(alphabet: List[str], sequence: List[str]) -> List[np.ndarray]:
    """Return Godel-encoded cylinders for sequence, applying all possible enumerations for alphabet symbols in encoding.

    :param alphabet: A list of symbols comprising an alphabet
    :param sequence: A list of symbols from the alphabet
    :return: encoded cylinders for sequence when applying Godel encodings with all possible enumerations, with each
        element of returned list being of the form np.ndarray([seq_left_bound, seq_right_bound])
    """
    cylinders = []
    for alpha_p in permutations(alphabet, len(alphabet)):
        cylinders.append(GodelEncoder(alpha_p).encode_cylinder(sequence))
    return list(sorted(cylinders, key=lambda x: x[0]))


def get_invariant_partition_2d(
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
    alpha_cyl = get_invariant_partition_1d(alpha_alphabet, alpha_sequence)
    beta_cyl = get_invariant_partition_1d(beta_alphabet, beta_sequence)
    partition = list(product(alpha_cyl, beta_cyl))
    return partition


