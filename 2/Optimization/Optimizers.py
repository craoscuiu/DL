import numpy as np
class Sgd:

    def __init__ (self,learning_rate ):
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor ):
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate ):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0


    def calculate_update(self,weight_tensor, gradient_tensor ):
        v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        self.v = v
        return weight_tensor + v


class Adam:

    def __init__(self, learning_rate, mu, rho):
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
        return weight_tensor - self.learning_rate * (v_corr /(np.sqrt(r_corr) + np.finfo(float).eps))