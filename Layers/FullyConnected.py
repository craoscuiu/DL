import numpy as np
from Layers import Base


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights_shape = (input_size, output_size)
        self.bias_shape = (1, output_size)
        # self.weights = np.random.uniform(0, 1, (self.weights_shape )
        # self.bias = np.ones(self.bias_shape)
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None
        self._gradient_weights = None
        self.initial_input = None
        self.updated_weights = None
        self.gradient_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1] = weights_initializer.initialize(self.weights_shape, self.weights_shape[0],
                                                           self.weights_shape[1])
        self.weights[-1] = bias_initializer.initialize(self.bias_shape, self.bias_shape[0], self.bias_shape[1])
        # self.bias = bias_initializer.initialize(self.bias_shape, self.bias_shape[0], self.bias_shape[1])

    def forward(self, input_tensor):
        # self.initial_input = input_tensor
        shape = input_tensor.shape
        ones = np.ones((shape[0], 1))
        self.initial_input = np.hstack((input_tensor, ones))
        # weight_bias = np.vstack((self.weights, self.bias ))
        return np.dot(self.initial_input, self.weights)

    def backward(self, error_tensor):
        weights = np.transpose(self.weights)
        error = np.dot(error_tensor, weights)

        self.gradient_tensor = np.dot(np.transpose(self.initial_input), error_tensor)
        self._gradient_weights = self.gradient_tensor
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_tensor)
        return error[:, :-1]

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_tensor):
        self._gradient_weights = gradient_tensor

    optimizer = property(get_optimizer, set_optimizer)
