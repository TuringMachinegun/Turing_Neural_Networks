__author__ = 'Giovanni Carmantini'

from simpleNNlib import *


class NeuralTM():
    """
    Given the parameters for the piecewise linear system implementing an NDA,
    constructs the equivalent neural network described in (OUR PAPER).
    
    :param x_params: The parameters for the x component of the linear transformation.
        the parameters are to be passed as a numpy ndarray with 4 dimensions.
        The first two dimensions are respectively the cell i,j partition coordinates,
         the third and fourth dimensions are respectively :math:`a^{i, j}_x, b^{i, j}_x`
         in the linear transformation  :math:`a^{i, j}_x x + b^{i, j}_x` of the cell
    :param y_params: The parameters for the y component of the linear transformation.
        the parameters are to be passed as a numpy ndarray with 4 dimensions.
        The first two dimensions are respectively the cell i,j partition coordinates,
         the third and fourth dimensions are respectively :math:`a^{i, j}_y, b^{i, j}_y`
         in the linear transformation  :math:`a^{i, j}_y y + b^{i, j}_y` of the cell
    """

    def __init__(self, x_params, y_params):

        self.x_params = x_params
        self.y_params = y_params

        max_linx = np.amax(self.x_params[:, :, 0] + self.x_params[:, :, 1])
        max_liny = np.amax(self.y_params[:, :, 0] + self.y_params[:, :, 1])
        self.h = np.amax((max_linx, max_liny))*2  # inhibitory constant

        self.x_nsymbols = x_params.shape[0]
        self.y_nsymbols = x_params.shape[1]
        self.nstates = self.x_nsymbols * self.y_nsymbols

        # construct input layer

        self.xinp_lr = RampLayer(2, initial_values=[0.25, 0.5])
        self.yinp_lr = RampLayer(2, initial_values=[0, 0.5])

        # construct cell selection layer

        self.xcellselect_lr = HeavisideLayer(self.x_nsymbols, centers=np.linspace(0, 1, self.x_nsymbols+1)[:-1],
                                             inclusive=[1] * self.x_nsymbols)
        Connection(self.xinp_lr, self.xcellselect_lr, [[1] * self.x_nsymbols, [0] * self.x_nsymbols])

        self.ycellselect_lr = HeavisideLayer(self.y_nsymbols, centers=np.linspace(0, 1, self.y_nsymbols+1)[:-1],
                                             inclusive=[1] * self.y_nsymbols)
        Connection(self.yinp_lr, self.ycellselect_lr, [[1] * self.y_nsymbols, [0] * self.y_nsymbols])

        # construct linear transformation layer

        self.lintransf_lr = RampLayer(self.nstates * 2 * 2,
                                      biases=self.lintransf_biases_from_params() - self.h)
        Connection(self.xcellselect_lr, self.lintransf_lr,
                   self.cn_xcellselect_lintransf())
        Connection(self.ycellselect_lr, self.lintransf_lr,
                   self.cn_ycellselect_lintransf())
        Connection(self.xinp_lr, self.lintransf_lr, self.cn_xinp_lintransf())
        Connection(self.yinp_lr, self.lintransf_lr, self.cn_yinp_lintransf())

        lintransf_xinp_cnmat = self.cn_lintransf_xinp()
        lintransf_yinp_cnmat = self.cn_lintransf_yinp()
        Connection(self.lintransf_lr, self.xinp_lr, lintransf_xinp_cnmat)
        Connection(self.lintransf_lr, self.yinp_lr, lintransf_yinp_cnmat)

    def cn_xcellselect_lintransf(self):
        """
        Generates the connection matrix between the x cell selection layer and the linear transformation layer.
        The connection pattern is discussed in (OUR PAPER).

        :return: connection matrix between x cell selection layer and linear transformation layer
        :rtype: numpy ndarray
        """
        x_part_mat = []
        for x_cell in range(self.x_nsymbols):  # there are x_nsymbols neurons in the x cell selection layer
            cell_part_mat = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))  # first put all connections to 0
            cell_part_mat[x_cell] = self.h/2  # then make the relevant ones equal to h/2
            if x_cell > 0:  # and the ones to the right of the current cell equal to -h/2
                cell_part_mat[x_cell-1] = -self.h/2
            x_part_mat.append(cell_part_mat.flatten())
        return np.array(x_part_mat)

    def cn_ycellselect_lintransf(self):
        """
        Generates the connection matrix between the y cell selection layer and the linear transformation layer.
        The connection pattern is discussed in (OUR PAPER).

        :return: connection matrix between y cell selection layer and linear transformation layer
        :rtype: numpy ndarray
        """

        alpha_part_mat = []
        for cell in range(self.y_nsymbols):  # there are n_beta neurons in the beta cell selection layer
            cell_part_mat = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))  # first put all connections to 0
            cell_part_mat[:, cell] = self.h/2  # then make the relevant ones equal to h/2
            if cell > 0:  # and the ones to the right of the current cell equal to -h/2
                cell_part_mat[:, cell-1] = -4
            alpha_part_mat.append(cell_part_mat.flatten())
        return np.array(alpha_part_mat)

    def cn_xinp_lintransf(self):
        """
        Generates the connection matrix between the x input neurons and the linear transformation layer.
        These are the connections implementing the linear transformations of the NDA piecewise linear system.
        The multiplication constants are implemented as multiplicative weights in the connection matrix,
        whereas the additive constants are implemented as biases to the linear transformation layer,
        outside this function.

        :return: connection matrix between x input neurons and linear transformation layer
        :rtype: numpy ndarray
        """
        conn_mat_x1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
        conn_mat_x2 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))

        conn_mat_x1[:, :, 0, 0] = self.x_params[:, :, 0]
        conn_mat_x2[:, :, 0, 1] = self.x_params[:, :, 0]

        return np.array([conn_mat_x1.flatten(), conn_mat_x2.flatten()])

    def cn_yinp_lintransf(self):
        """
        Generates the connection matrix between the y input neurons and the linear transformation layer.
        These are the connections implementing the linear transformations of the NDA piecewise linear system.
        The multiplication constants are implemented as multiplicative weights in the connection matrix,
        whereas the additive constants are implemented as biases to the linear transformation layer,
        outside this function.

        :return: connection matrix between y input neurons and linear transformation layer
        :rtype: numpy ndarray
        """

        conn_mat_x1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
        conn_mat_x2 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))

        conn_mat_x1[:, :, 1, 0] = self.y_params[:, :, 0]
        conn_mat_x2[:, :, 1, 1] = self.y_params[:, :, 0]

        return np.array([conn_mat_x1.flatten(), conn_mat_x2.flatten()])

    def lintransf_biases_from_params(self):
        """
        Generates the biases for the neurons in the linear transformation layer, which implement the additive constants
        of the linear transfomations of the NDA piecewise linear system.

        :return: biases for linear transformation layer neurons
        :rtype: numpy ndarray
        """

        biases = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
        biases[:, :, 0, 0] = self.x_params[:, :, 1]
        biases[:, :, 0, 1] = self.x_params[:, :, 1]
        biases[:, :, 1, 0] = self.y_params[:, :, 1]
        biases[:, :, 1, 1] = self.y_params[:, :, 1]
        return biases.flat

    def cn_lintransf_xinp(self):
        """
        :return: connection matrix between linear transformation layer and x input neurons.
        :rtype: numpy ndarray
        """

        to_x_0 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
        to_x_0[:, :, 0, 0] = 1
        to_x_1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
        to_x_1[:, :, 0, 1] = 1

        x_out_conn = np.array([to_x_0.flatten(), to_x_1.flatten()])

        return x_out_conn.T

    def cn_lintransf_yinp(self):
        """
        :return: connection matrix between linear transformation layer and x input neurons.
        :rtype: numpy ndarray
        """

        to_y_0 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
        to_y_0[:, :, 1, 0] = 1
        to_y_1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
        to_y_1[:, :, 1, 1] = 1

        y_out_conn = np.array([to_y_0.flatten(), to_y_1.flatten()])

        return y_out_conn.T

    def run_net(self, init_x, init_y, n_iterations):
        """
        Runs the network, given the initial values for the x and y input neurons and the number of iterations

        :param init_x: initial activation values for left and right x input neurons
        :param init_y: initial activation values for left and right y input neurons
        :param n_iterations: the desired number of network iterations

        :return: list containing the input activation for each iteration. Each tuple item in the list contains
            two arrays, respectively containing the activations for the x and the y input neurons.
            ``[ (np.array([x_l_act0, x_r_act0]), np.array([y_l_act0, y_r_act0])), ... ]``
        :rtype: list
        """

        act_history = []

        self.xinp_lr.activation = np.array(init_x)
        self.yinp_lr.activation = np.array(init_y)

        act_history.append((self.xinp_lr.activation, self.yinp_lr.activation))

        for iteration in range(n_iterations):
            self.xcellselect_lr.activate()
            self.ycellselect_lr.activate()
            self.lintransf_lr.activate()
            self.xinp_lr.activate()
            self.yinp_lr.activate()

            act_history.append((self.xinp_lr.activation, self.yinp_lr.activation))

        return act_history
