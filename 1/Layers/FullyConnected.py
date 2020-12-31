import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size):
        self.bias = np.ones((1, output_size ))
        self.weights = np.random.rand(input_size, output_size )
        self._optimizer = None
        self._gradient_weights = None
        self.initial_input = None
        self.updated_weights = None
        self.gradient_tensor = None


    def forward(self,input_tensor):
        self.initial_input = input_tensor
        shape = input_tensor.shape
        ones = np.ones((shape[0],1))
        input_bias = np.hstack( (input_tensor, ones) )
        weight_bias = np.vstack((self.weights, self.bias ))
        return np.dot( input_bias , weight_bias )

    def backward(self,error_tensor):
        weights = np.transpose(self.weights)
        error = np.dot(error_tensor, weights)

        self.gradient_tensor = np.dot(np.transpose(self.initial_input), error_tensor)
        self._gradient_weights = self.gradient_tensor
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_tensor)
        else:
            print("Obo")
        return error


    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_tensor ):
        self._gradient_weights = gradient_tensor

    optimizer = property(get_optimizer, set_optimizer)
