__author__ = 'Giovanni Carmantini'

import simpleNNlib as snn
import numpy as np

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

    def __init__(self, x_params, y_params, cylinder_sets=False):
        
        self.cylinder_sets = cylinder_sets
        
        if self.cylinder_sets:
            self.d_factor = 2
        else:
            self.d_factor = 1
        
        self.x_params = x_params
        self.y_params = y_params

        max_linx = np.amax(self.x_params[:, :, 0] + self.x_params[:, :, 1])
        max_liny = np.amax(self.y_params[:, :, 0] + self.y_params[:, :, 1])
        self.h = np.amax((max_linx, max_liny))*2  # inhibitory constant

        self.x_nsymbols = x_params.shape[0]
        self.y_nsymbols = x_params.shape[1]
        self.nstates = self.x_nsymbols * self.y_nsymbols

        # construct input layer

        self.MCLx = snn.RampLayer(self.d_factor, initial_values=[0.0]*self.d_factor)
        self.MCLy = snn.RampLayer(self.d_factor, initial_values=[0.0]*self.d_factor)

        # construct cell selection layer

        self.BSLbx = snn.HeavisideLayer(self.x_nsymbols, 
                                                 centers=np.linspace(0, 1, self.x_nsymbols+1)[:-1],
                                                 inclusive=[1] * self.x_nsymbols)
        self.BSLby = snn.HeavisideLayer(self.y_nsymbols,
                                                 centers=np.linspace(0, 1, self.y_nsymbols+1)[:-1],
                                                 inclusive=[1] * self.y_nsymbols)
        
        if self.cylinder_sets:
            snn.Connection(self.MCLx, self.BSLbx, 
                           [[1] * self.x_nsymbols, [0] * self.x_nsymbols])
            snn.Connection(self.MCLy, self.BSLby, 
                           [[1] * self.y_nsymbols, [0] * self.y_nsymbols])
        else:
            snn.Connection(self.MCLx, self.BSLbx, 
                           np.array([[1] * self.x_nsymbols]))
            snn.Connection(self.MCLy, self.BSLby, 
                           np.array([[1] * self.y_nsymbols]))

        # construct linear transformation layer

        if self.cylinder_sets:
            self.LTL = snn.RampLayer(self.nstates * 2 * 2,
                                     biases=self.LTL_biases_from_params() 
                                     - self.h)
        else:
            self.LTL = snn.RampLayer(self.nstates * 2,
                                     biases=self.LTL_biases_from_params() 
                                     - self.h)
            
        
        BSLbx_LTL_cnmat = self.cn_BSLbx_LTL()
        BSLby_LTL_cnmat = self.cn_BSLby_LTL()
        snn.Connection(self.BSLbx, self.LTL, BSLbx_LTL_cnmat)
        snn.Connection(self.BSLby, self.LTL, BSLby_LTL_cnmat)
       
        MCLx_LTL_cnmat = self.cn_MCLx_LTL()
        MCLy_LTL_cnmat = self.cn_MCLy_LTL()
        snn.Connection(self.MCLx, self.LTL, MCLx_LTL_cnmat)
        snn.Connection(self.MCLy, self.LTL, MCLy_LTL_cnmat)

        LTL_MCLx_cnmat = self.cn_LTL_MCLx()
        LTL_MCLy_cnmat = self.cn_LTL_MCLy()
        snn.Connection(self.LTL, self.MCLx, LTL_MCLx_cnmat)
        snn.Connection(self.LTL, self.MCLy, LTL_MCLy_cnmat)

    def cn_BSLbx_LTL(self):
        """
        Generates the connection matrix between the x cell selection layer and the linear transformation layer.
        The connection pattern is discussed in (OUR PAPER).

        :return: connection matrix between x cell selection layer and linear transformation layer
        :rtype: numpy ndarray
        """

        x_part_mat = []
        for cells_col in range(self.x_nsymbols):  # there are x_nsymbols neurons in the x cell selection layer

            if self.cylinder_sets:
                cell_cn_mat = np.zeros((self.x_nsymbols, 
                                          self.y_nsymbols, 2, 2))  # first put all connections to 0
            else:
                cell_cn_mat = np.zeros((self.x_nsymbols, 
                                          self.y_nsymbols, 2))  # first put all connections to 0

            cell_cn_mat[cells_col] = self.h/2  # then make the relevant ones equal to h/2
            if cells_col > 0:  # and the ones to the right of the current cell equal to -h/2
                cell_cn_mat[cells_col-1] = -self.h/2
            x_part_mat.append(cell_cn_mat.flatten())
        return np.array(x_part_mat)

    def cn_BSLby_LTL(self):
        """
        Generates the connection matrix between the y cell selection layer and the linear transformation layer.
        The connection pattern is discussed in (OUR PAPER).

        :return: connection matrix between y cell selection layer and linear transformation layer
        :rtype: numpy ndarray
        """

        y_part_mat = []
        for cells_row in range(self.y_nsymbols):  # there are n_beta neurons in the beta cell selection layer
           
            if self.cylinder_sets:
                cell_cn_mat = np.zeros((self.x_nsymbols, 
                                          self.y_nsymbols, 2, 2))  # first put all connections to 0
            else:
                cell_cn_mat = np.zeros((self.x_nsymbols,
                                          self.y_nsymbols, 2))  # first put all connections to 0

            cell_cn_mat[:, cells_row] = self.h/2  # then make the relevant ones equal to h/2
            if cells_row > 0:  # and the ones to the right of the current cell equal to -h/2
                cell_cn_mat[:, cells_row - 1] = -self.h/2
            y_part_mat.append(cell_cn_mat.flatten())

        return np.array(y_part_mat)

    def cn_MCLx_LTL(self):
        """
        Generates the connection matrix between the x input neurons and the linear transformation layer.
        These are the connections implementing the linear transformations of the NDA piecewise linear system.
        The multiplication constants are implemented as multiplicative weights in the connection matrix,
        whereas the additive constants are implemented as biases to the linear transformation layer,
        outside this function.

        :return: connection matrix between x input neurons and linear transformation layer
        :rtype: numpy ndarray
        """
        if self.cylinder_sets:
            conn_mat_x1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            conn_mat_x2 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            conn_mat_x1[:, :, 0, 0] = self.x_params[:, :, 0]
            conn_mat_x2[:, :, 0, 1] = self.x_params[:, :, 0]

            conn_mat = np.array([conn_mat_x1.flatten(), conn_mat_x2.flatten()])
        
        else:
            conn_mat_x = np.zeros((self.x_nsymbols, self.y_nsymbols, 2))
            conn_mat_x[:, :, 0] = self.x_params[:, :, 0]

            conn_mat = np.array(conn_mat_x.flatten())


        return conn_mat

    def cn_MCLy_LTL(self):
        """
        Generates the connection matrix between the y input neurons and the linear transformation layer.
        These are the connections implementing the linear transformations of the NDA piecewise linear system.
        The multiplication constants are implemented as multiplicative weights in the connection matrix,
        whereas the additive constants are implemented as biases to the linear transformation layer,
        outside this function.

        :return: connection matrix between y input neurons and linear transformation layer
        :rtype: numpy ndarray
        """
        if self.cylinder_sets:
            conn_mat_y1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            conn_mat_y2 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            conn_mat_y1[:, :, 1, 0] = self.y_params[:, :, 0]
            conn_mat_y2[:, :, 1, 1] = self.y_params[:, :, 0]

            conn_mat = np.array([conn_mat_y1.flatten(), conn_mat_y2.flatten()])


        else:
            conn_mat_y = np.zeros((self.x_nsymbols, self.y_nsymbols, 2))
            conn_mat_y[:, :, 1] = self.y_params[:, :, 0]

            conn_mat = np.array(conn_mat_y.flatten())
        return conn_mat

    def LTL_biases_from_params(self):
        """
        Generates the biases for the neurons in the linear transformation layer, which implement the additive constants
        of the linear transfomations of the NDA piecewise linear system.

        :return: biases for linear transformation layer neurons
        :rtype: numpy ndarray
        """

        if self.cylinder_sets:
            biases = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            biases[:, :, 0, 0] = self.x_params[:, :, 1]
            biases[:, :, 0, 1] = self.x_params[:, :, 1]
            biases[:, :, 1, 0] = self.y_params[:, :, 1]
            biases[:, :, 1, 1] = self.y_params[:, :, 1]
        
        else:
            biases = np.zeros((self.x_nsymbols, self.y_nsymbols, 2))
            biases[:, :, 0] = self.x_params[:, :, 1]
            biases[:, :, 1] = self.y_params[:, :, 1]


        return biases.flat

    def cn_LTL_MCLx(self):
        """
        :return: connection matrix between linear transformation layer and x input neurons.
        :rtype: numpy ndarray
        """

        if self.cylinder_sets:
            to_x_0 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            to_x_0[:, :, 0, 0] = 1
            to_x_1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            to_x_1[:, :, 0, 1] = 1
            x_out_conn = np.array([to_x_0.flatten(), to_x_1.flatten()]).T
        
        else:
            to_x = np.zeros((self.x_nsymbols, self.y_nsymbols, 2))
            to_x[:, :, 0] = 1
            x_out_conn = np.array(to_x.flatten())[:,None]


        return x_out_conn

    def cn_LTL_MCLy(self):
        """
        :return: connection matrix between linear transformation layer and x input neurons.
        :rtype: numpy ndarray
        """
        if self.cylinder_sets:
            to_y_0 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            to_y_0[:, :, 1, 0] = 1
            to_y_1 = np.zeros((self.x_nsymbols, self.y_nsymbols, 2, 2))
            to_y_1[:, :, 1, 1] = 1

            y_out_conn = np.array([to_y_0.flatten(), to_y_1.flatten()]).T
        
        else:
            to_y = np.zeros((self.x_nsymbols, self.y_nsymbols, 2))
            to_y[:, :, 1] = 1

            y_out_conn = np.array(to_y.flatten())[:,None]


        return y_out_conn

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

        self.MCLx.activation[:] = np.array(init_x)
        self.MCLy.activation[:] = np.array(init_y)

        if self.cylinder_sets:
            MCL_acts = (self.MCLx.activation, self.MCLy.activation)
        else:
            MCL_acts = (self.MCLx.activation[0], self.MCLy.activation[0])
            
        act_history.append(MCL_acts)

        for iteration in range(n_iterations):
            self.BSLbx.activate()
            self.BSLby.activate()
            self.LTL.activate()
            self.MCLx.activate()
            self.MCLy.activate()

            if self.cylinder_sets:
                MCL_acts = (self.MCLx.activation, self.MCLy.activation)
            else:
                MCL_acts = (self.MCLx.activation[0], self.MCLy.activation[0])

            act_history.append(MCL_acts)
                

        return act_history
