'''
import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.gradient_tensor = None
        self.initial_input = None


    def forward(self,input_tensor, label_tensor ):
        self.initial_input = input_tensor
        self.similarity = 1- np.absolute(input_tensor - label_tensor )
        self.similarity = self.similarity + np.finfo(float).eps
        loss_matrix = -1 * np.log(self.similarity)
        loss_matrix = np.where((label_tensor == 1), loss_matrix, 0 )
        return np.sum(loss_matrix)

    def backward(self, label_tensor):
        self.gradient_tensor = -1 * label_tensor / ( self.initial_input + np.finfo(float).eps )
        self.gradient_tensor= np.where((label_tensor == 1), self.gradient_tensor, 0)
        return self.gradient_tensor





'''
import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.gradient_tensor = None
        self.initial_input = None

    def forward(self,input_tensor, label_tensor):
        self.initial_input = input_tensor
        loss_matrix = -1 * np.log(input_tensor + np.finfo(float).eps)
        loss_matrix = np.where((label_tensor == 1), loss_matrix, 0)
        return np.sum(loss_matrix)

    def backward(self, label_tensor):
        self.gradient_tensor = -1 * label_tensor / self.initial_input
        return self.gradient_tensor