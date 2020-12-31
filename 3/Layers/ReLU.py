import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()
        self.gradient_tensor = None
        self.initial_input = None



    def forward(self,input_tensor):
        self.initial_input = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        self.gradient_tensor = error_tensor * np.heaviside( self.initial_input,1)
        return self.gradient_tensor

