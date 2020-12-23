import numpy as np
from Layers import Base


class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.batch = None
        self.size = None
        #pass

    def forward(self, input_tensor):
        self.size = input_tensor.shape[1:]
        self.batch = input_tensor.shape[0]
        return input_tensor.reshape((self.batch, np.prod(self.size)))

    def backward(self, error_tensor):
        size = (self.batch, ) + self.size
        return np.reshape(error_tensor, size)
