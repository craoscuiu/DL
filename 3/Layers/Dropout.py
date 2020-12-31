from Layers.Base import BaseLayer
import numpy as np
from copy import deepcopy
class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        input = deepcopy(input_tensor)
        if self.testing_phase:
            out = input
        else:
            self.mask = np.random.binomial(1, self.probability, size=input.shape) / self.probability
            out = input * self.mask

        return out


    def backward(self, error_tensor):
        gradient = error_tensor * self.mask

        return gradient
