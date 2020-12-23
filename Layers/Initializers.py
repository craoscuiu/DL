import numpy as np
from Layers import Base


class Constant(Base.BaseLayer):
    def __init__(self, constant=0.1):
        super().__init__()
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant)


class UniformRandom(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, weights_shape)


class Xavier(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)


class He(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)