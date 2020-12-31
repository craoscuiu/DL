import numpy as np
class Flatten:
    def __init___(self):
        self.batch = None
        self.size = None
        pass

    def forward(self, input_tensor):
        self.batch = input_tensor.shape[0]
        self.size = input_tensor.shape[1:]
        return input_tensor.reshape(self.batch, np.prod(self.size))

    def backward(self, error_tensor):
        size = (self.batch, ) + self.size
        return np.reshape(error_tensor, size)
