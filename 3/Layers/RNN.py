import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH
import NeuralNetwork
from copy import deepcopy
from Optimization import Optimizers, Constraints

class RNN(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weights_initializer = None
        self.bias_initializer = None
        self.weights_shape = (input_size + hidden_size + 1,hidden_size)
        self.hidden_state = None
        self._memorize = False
        self._optimizer = Optimizers.Sgd(0)
        self._optimizer.add_regularizer(Constraints.L2_Regularizer(0))
        self._gradient_weights = np.zeros(self.weights_shape)
        self._weights = None
        self.FC2_weights = None
        self.layers = list()

    def initialize(self, weights_initializer, bias_initializer):
        self._weights = weights_initializer.initialize(self.weights_shape, self.weights_shape[0],
                                                           self.weights_shape[1])
        self._weights[-1] = bias_initializer.initialize((1,1),1,1)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self,input_tensor):
       output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
       if self.weights_initializer is not None and self.FC2_weights is None:
           self.FC2_weights = self.weights_initializer.initialize((self.hidden_size + 1,self.output_size),
                                                             self.hidden_size + 1, self.output_size)
           self.FC2_weights[-1] = self.bias_initializer.initialize((1, 1), 1, 1)


       if not self._memorize or self.hidden_state is None:
           self.hidden_state = np.zeros(self.hidden_size)


       for i in range(input_tensor.shape[0]):

            x_tilde = np.concatenate((self.hidden_state, input_tensor[i] ))

            FC1 = FullyConnected(x_tilde.size, self.hidden_state.size )
            TanH_FC1 = TanH()
            FC2 = FullyConnected(self.hidden_state.size, self.output_size)
            Sigmoid_FC2 = Sigmoid()

            if self._weights is None:
                self._weights = FC1.weights
            else:
                FC1.weights = self._weights

            if self.FC2_weights is None:
                self.FC2_weights = FC2.weights
            else:
                FC2.weights = self.FC2_weights


            output = TanH_FC1.forward(FC1.forward(x_tilde))
            self.hidden_state = deepcopy(output)
            output_tensor[i] = Sigmoid_FC2.forward(FC2.forward(output))
            if len(self.layers) < 4 * input_tensor.shape[0]:
                self.layers.extend((FC1, TanH_FC1, FC2, Sigmoid_FC2))
            else:
                self.layers[4*i] = FC1
                self.layers[4*i+1] = TanH_FC1
                self.layers[4*i+2] = FC2
                self.layers[4*i+3] = Sigmoid_FC2


       if self._optimizer.regularizer:
        reg_loss = self._optimizer.calculate_regularization_loss(self._weights)
       else:
        reg_loss = 0

       return output_tensor + reg_loss

    def backward(self, error_tensor):
        output_tensor = np.zeros((error_tensor.shape[0], self.input_size))
        error_hidden = np.zeros((1,self.hidden_size))

        self._gradient_weights = np.zeros(self.weights_shape)

        for i in reversed(range(error_tensor.shape[0])):

            Sigmoid_FC2 = self.layers[4*i+3]
            FC2 = self.layers[4*i+2]
            error_gradient = FC2.backward(Sigmoid_FC2.backward(error_tensor[i]))

            error_gradient += error_hidden

            TanH_FC1 = self.layers[4*i+1]
            FC1 = self.layers[4*i]
            error_gradient = FC1.backward(TanH_FC1.backward(error_gradient))

            output_tensor[i] = error_gradient[0, self.hidden_size : ]
            error_hidden = error_gradient[0, :self.hidden_size]
            self._gradient_weights += FC1.gradient_weights

        self._weights = self._optimizer.calculate_update(self._weights, self._gradient_weights)

        for layer in self.layers[::4]:
            layer.weights = self._weights


        return output_tensor


    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights