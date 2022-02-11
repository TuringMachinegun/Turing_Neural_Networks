__author__ = "Giovanni Sirio Carmantini"
__version__ = 0.1

import numpy as np
from collections import defaultdict


class AbstractNNLayer(object):
    """
    The base class for all neuron layers. Each layer has to implement the possibility to
    add connections from other layers,
    to compute the input to each neuron by summing all contribution from it connections,
    and to compute the output given the input and some activation function.

    In particular, derived classes will override the activation funcion with their own.
    For example, a ramp layer will use a ramp activation function,
    a heaviside layer will use a heaviside function, etc...

    :param n_units: the number of units in the layer
    :param initial_values: a list of activation values with which the units will be
        initialized
    """

    def __init__(self, n_units, initial_values=[]):
        self.n_units = n_units
        self.connections = []
        if not initial_values:
            self.activation = np.zeros(n_units)
        else:
            self.activation = np.array(initial_values)
            if self.activation.size != n_units:
                raise ValueError("initializing with wrong number of values.")
        self.transfer_function = (
            None  # this is the abstract class, need to implement a specific one
        )

    def _add_connection(self, from_layer, connection_matrix):
        """
        Add a connection from from_layer to this layer, with weights as in connection_matrix.
        Return the connection matrix, so that it is possible to edit it subsequently.
        :warning: don't use this function directly, rely on Connection class to establish connections.
        """
        conn_mat = np.array(connection_matrix)
        self.connections.append((from_layer, conn_mat))
        return conn_mat

    def sum_input(self):
        """Computes the input contribution from connected layers."""
        running_sum = np.zeros(self.n_units)
        if self.connections:
            for from_layer, connection_matrix in self.connections:
                if from_layer.activation.ndim == connection_matrix.ndim == 1:
                    running_sum += from_layer.activation * connection_matrix
                else:
                    running_sum += np.dot(from_layer.activation, connection_matrix)
        return running_sum

    def activate(self):
        """Computes the units activation."""
        self.activation = self.transfer_function(self.sum_input())

    def __getitem__(self, item):
        return LayerSlice(self, item)  # returns activation of item


class HeavisideLayer(AbstractNNLayer):
    """
    Class implementing a layer of neurons with Heaviside activation function.

    :param n_units: number of units in layer.
    :param centers: specifies :math:`c_i` for each unit :math:`i` , given the unit activation function :math:`H_i(x-c_i)` .
        Equivalent to adding a bias of :math:`-c_i` .
    :param inclusive: if :math:`H_i(-c_i)=1`, the :math:`i`-th item in the list is True,
        otherwise it's False and :math:`H_i(-c_i)=0`.
    """

    def __init__(self, n_units, centers, inclusive=[]):
        AbstractNNLayer.__init__(self, n_units)
        self.centers = np.array(centers)
        self.inclusive = (
            np.array(inclusive, dtype=bool)
            if inclusive
            else np.zeros(n_units, dtype=bool)
        )
        if not self.centers.size == self.inclusive.size == self.n_units:
            raise ValueError("Array must be of the same size as layer.")
        self.exclusive = np.invert(self.inclusive)

    def activate(self):
        """Computes the units activation"""
        inp_sum = self.sum_input()
        self.activation[self.inclusive] = (
            inp_sum[self.inclusive] >= self.centers[self.inclusive]
        ).astype(int)
        self.activation[self.exclusive] = (
            inp_sum[self.exclusive] > self.centers[self.exclusive]
        ).astype(int)
        return self.activation


class RampLayer(AbstractNNLayer):
    """
    Class implementing a layer of neurons with saturated linear activation function.

    :param n_units: number of units in layer.
    :param biases: a list specifying the bias for each neuron.
    :param initial_values: initial activation values
    """

    def __init__(self, n_units, biases=0, initial_values=[]):
        AbstractNNLayer.__init__(self, n_units, initial_values)
        self.biases = biases

    def activate(self):
        self.activation = np.clip(self.sum_input() + self.biases, 0, 1)
        return self.activation


class LayerSlice:
    """
    This class is used so that to access the activation for a given layer *nnlayer*,
    you can just use the notation *nnlayer[someslice]*, where *someslice* is the usual Python slice.
    """

    def __init__(self, layer, a_slice):
        self.layer = layer
        self.a_slice = a_slice

    def get_activation(self):
        return self.layer.activation[self.a_slice]

    activation = property(get_activation)


class Connection:
    """
    Class implementing a connection. Basically used to keep track of who is connected to who at this moment.
    Also, you can modify the connection by modifying conn_mat.

    :param from_layer: layer from which the connection is established.
    :param to_layer: layer to which the connection is established.
    :param connection_matrix: matrix specifying connections and weights.
    """

    # I don't remember why I wanted to keep track of all connections, I don't use this anywhere
    keep_track = defaultdict(list)

    def __init__(self, from_layer, to_layer, connection_matrix):
        self.sender = from_layer
        self.receiver = to_layer
        self.conn_mat = to_layer._add_connection(from_layer, connection_matrix)
        Connection.keep_track[to_layer].append(from_layer)
