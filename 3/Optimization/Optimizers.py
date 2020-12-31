import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_regularization_loss(self, weights):
        return self.regularizer.norm(weights)

class Sgd(Optimizer):

    def __init__ (self,learning_rate ):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor ):
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0


    def calculate_update(self,weight_tensor, gradient_tensor ):
        v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        self.v = v
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor + v


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
        v = self.mu * self.v + (1 - self.mu )*gradient_tensor
        r = self.rho * self.r + ( 1 - self.rho ) * gradient_tensor * gradient_tensor
        v_corr = v / (1 - np.power(self.mu, self.exponent))
        r_corr = r / (1 - np.power(self.rho, self.exponent))

        self.v = v
        self.r = r
        self.exponent = self.exponent + 1
        #weighted momentum in different dimensions
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * (v_corr /(np.sqrt(r_corr) + np.finfo(float).eps))

