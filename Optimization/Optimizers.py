import numpy as np


class Optimizer:

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        shrinkage = 0
        if self.regularizer:
            shrinkage = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return weight_tensor - shrinkage - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        shrinkage = 0
        if self.regularizer:
            shrinkage = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        self.v = v
        return weight_tensor - shrinkage + v


class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.exponent = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        shrinkage = 0
        if self.regularizer:
            shrinkage = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor
        v_corr = v / (1 - np.power(self.mu, self.exponent))
        r_corr = r / (1 - np.power(self.rho, self.exponent))

        self.v = v
        self.r = r
        self.exponent = self.exponent + 1
        # weighted momentum in different dimensions
        return weight_tensor - shrinkage - self.learning_rate * (v_corr / (np.sqrt(r_corr) + np.finfo(float).eps))
