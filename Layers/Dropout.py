from Layers import Base
import numpy as np


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None
        self._optimizer = None

    def forward(self, input_tensor):
        if self.testing_phase:
            input_tensor = input_tensor
        else:
            self.mask = np.random.binomial(1, p=self.probability, size=input_tensor.shape)  # binomial converts probability into 1-p
            input_tensor = (1 / self.probability) * input_tensor * self.mask

        return input_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor * self.mask

        return error_tensor / self.probability
