import numpy as np
from Layers.Base import BaseLayer
class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.gradient_tensor = None
        self.initial_input = None
        self.y_hat = None

    def forward(self, input_tensor):
        self.initial_input = input_tensor
        batch_max = np.amax(input_tensor, 1)
        batch_max = batch_max.reshape(input_tensor.shape[0], 1)

        input_tensor = input_tensor - batch_max
        input_tensor = np.exp(input_tensor)

        batch_sum = input_tensor.sum(1)
        batch_sum = batch_sum.reshape(input_tensor.shape[0], 1)

        self.y_hat = input_tensor/batch_sum

        return self.y_hat

    def backward(self, error_tensor):
        sum_parts = error_tensor * self.y_hat
        sum_parts = sum_parts.sum(1)
        sum_parts = sum_parts.reshape(error_tensor.shape[0],1)

        self.gradient_tensor = self.y_hat * ( error_tensor - sum_parts )
        return self.gradient_tensor