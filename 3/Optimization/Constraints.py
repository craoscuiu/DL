import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return weights * self.alpha

    def norm(self, weights):
        return self.alpha * np.square(np.linalg.norm(weights))

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return np.sign(weights) * self.alpha

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))