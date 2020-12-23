import numpy as np
from Layers import Base, Helpers
from Optimization import Optimizers
from copy import deepcopy


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self._optimizer = None
        self.channels = channels

        self.bias, self._gradient_bias = None, None
        self.weights, self._gradient_weights = None, None
        BatchNormalization.initialize(self)

        self.input_tensor = None
        self.mean_tilde, self.mean_batch = None, None
        self.var_tilde, self.var_batch = None, None
        self.x_tilde = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.bias = np.zeros((1, self.channels))
        self.weights = np.ones((1, self.channels))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # reformat image2vec
        if len(self.input_tensor.shape) == 4:
            input_tensor = BatchNormalization.reformat(self, input_tensor)

        if self.testing_phase:
            # input normalization
            x_tilde = (input_tensor - self.mean_tilde) / np.sqrt(self.var_tilde + np.finfo(float).eps)
            y_hat = x_tilde * self.weights + self.bias
        else:
            batch_mean = np.mean(input_tensor, axis=0)
            batch_var = np.var(input_tensor, axis=0)
            self.mean_batch = np.expand_dims(np.array(batch_mean), axis=0)
            self.var_batch = np.expand_dims(np.array(batch_var), axis=0)

            # moving average estimation
            if self.mean_tilde is None:
                self.mean_tilde = self.mean_batch
                self.var_tilde = self.var_batch
            else:
                self.mean_tilde = 0.8*self.mean_tilde + 0.2*self.mean_batch
                self.var_tilde = 0.8*self.var_tilde + 0.2*self.var_batch

            # input normalization
            x_tilde = (input_tensor - self.mean_batch) / np.sqrt(self.var_batch + np.finfo(float).eps)
            y_hat = x_tilde*self.weights + self.bias
            self.x_tilde = x_tilde

        # reformat image2vec
        if len(self.input_tensor.shape) == 4:
            y_hat = BatchNormalization.reformat(self, y_hat)
            self.x_tilde = BatchNormalization.reformat(self, self.x_tilde)
        return y_hat

    def backward(self, error_tensor):
        if len(self.input_tensor.shape) == 4:  # reformat image2vec
            x_tilde = BatchNormalization.reformat(self, self.x_tilde)
            error_tensor = BatchNormalization.reformat(self, error_tensor)
            input_tensor = BatchNormalization.reformat(self, self.input_tensor)
        else:
            x_tilde = self.x_tilde
            input_tensor = self.input_tensor
        # gradient with respect to weights
        gradient_weights = np.expand_dims(np.sum(error_tensor*x_tilde, axis=0), axis=0)
        gradient_bias = np.expand_dims(np.sum(error_tensor, axis=0), axis=0)
        self._gradient_weights = deepcopy(gradient_weights)
        self._gradient_bias = deepcopy(gradient_bias)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)

        # gradient with respect to input
        error_tensor_up = Helpers.compute_bn_gradients(error_tensor, input_tensor, self.weights, self.mean_batch, self.var_batch)
        # reformat vec2img
        if len(self.input_tensor.shape) == 4:
            error_tensor_up = BatchNormalization.reformat(self, error_tensor_up)
        return error_tensor_up

    def reformat(self, tensor):
        if len(tensor.shape) == 2:  # tensor.shape = b*m*n x h
            b, h, m, n = self.input_tensor.shape
            tensor = tensor.reshape((b, m*n, h))
            tensor = np.transpose(tensor, (0, 2, 1)).reshape((b, h, m, n))
        elif len(tensor.shape) == 4:
            b, h, m, n = tensor.shape
            tensor = tensor.reshape((b, h, m*n))
            tensor = np.transpose(tensor, (0, 2, 1)).reshape((b*m*n, h))
        return tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)
