from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initiliazer, bias_initializer ):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        self.weights_initializer = weights_initiliazer
        self.bias_initializer = bias_initializer

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        output_tensor = self.loss_layer.forward(input_tensor, label_tensor)
        return output_tensor

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)



    def append_trainable_layer(self, layer):
        layer._optimizer = deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())

            self.backward()

    def test(self, input_tensor ):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        predictions = input_tensor
        return predictions
