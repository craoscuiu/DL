from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer ):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None
        self.default_weights = None

    def forward(self):
        reg_loss = 0
        input_tensor, label_tensor = self.data_layer.next()

        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)

        output_tensor = self.loss_layer.forward(input_tensor, label_tensor) + reg_loss
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
        self.phase = False
        for layer in self.layers:
            layer.testing_phase = self.phase

        for i in range(iterations):
            self.loss.append(self.forward())
            print("Loss: ", self.loss[i])
            self.backward()

    def test(self, input_tensor ):
        self.phase = True
        for layer in self.layers:
            layer.testing_phase = self.phase
            input_tensor = layer.forward(input_tensor)

        predictions = input_tensor
        return predictions

    def save(filename, net):
        with open(filename, 'wb') as f:
            pickle.dump(net, f)

    def load(filename, data_layer):
        with open(filename, 'rb') as f:
            net = pickle.load(f)
        net.data_layer = data_layer
        return net

    # def __getstate__():
    # copy(self.__dict__)
    # löschen data_layer, label_tensor
    # return

    # def __setstate__(state):
    # if state:
    # data_layer = None # auch label_tensor, self.__dict__.update(state)
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase