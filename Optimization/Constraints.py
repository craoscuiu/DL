import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        shrinkage = self.alpha * weights
        return shrinkage

    def norm(self, weights):
        if len(weights.shape) == 4:
            weights = weights.reshape((weights.shape[0], np.prod(weights.shape[1:])))
        norm_weights = self.alpha * (np.linalg.norm(weights, 'fro') ** 2)
        return norm_weights


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        shrinkage = self.alpha * np.sign(weights)
        return shrinkage

    def norm(self, weights):
        norm_weights = self.alpha * np.sum(np.abs(weights))
        return norm_weights
