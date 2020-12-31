from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
from Optimization import Optimizers
import numpy as np


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.initialize(self)
        self.optimizer = Optimizers.Sgd(0)
        self.alpha = 1
        self.mean, self.mean_k, self.var, self.var_k = None,None,None, None
        self.CNN = False

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(input_tensor.shape ) == 4:
            input_tensor = self.reformat(input_tensor)
            self.CNN = True

        if self.testing_phase:
            if self.mean_k is None:
                self.mean_k = self.mean
                self.var_k = self.var
            else:
                self.mean_k = self.mean_k * self.alpha + (1 - self.alpha) * self.mean
                self.var_k = self.var_k * self.alpha + (1 - self.alpha) * self.var

            self.normed_input = (input_tensor - self.mean_k) / np.sqrt(self.var_k + np.finfo(float).eps)
        else:
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            self.normed_input = (input_tensor - self.mean) / np.sqrt(self.var + np.finfo(float).eps)


        output = np.multiply(self.normed_input, self.weights ) + self.bias
        if self.CNN:
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        if len(error_tensor.shape ) == 4:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor)
        else:
            input_tensor = self.input_tensor

        error = compute_bn_gradients(error_tensor, input_tensor, self.weights, self.mean, self.var)

        self.gradient_bias = np.sum(error_tensor, axis = 0)
        self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias )

        self.gradient_weights = np.sum(error_tensor * self.normed_input, axis = 0)
        self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights )

        if self.CNN:
            error = self.reformat(error)

        return error

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            reshaped_tensor = tensor.reshape( *tensor.shape[:-2], -1 )
            reshaped_tensor = np.transpose(reshaped_tensor, (0,2,1))
            reshaped_tensor = reshaped_tensor.reshape(-1, reshaped_tensor.shape[-1])
        elif len(tensor.shape) == 2:
            shape = self.input_tensor.shape
            reshaped_tensor = tensor.reshape(shape[0], -1, shape[1])
            reshaped_tensor = np.transpose(reshaped_tensor, (0,2,1))
            reshaped_tensor = reshaped_tensor.reshape(shape)

        return reshaped_tensor

