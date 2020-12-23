class BaseLayer:
    def __init__(self, testing_phase=False):
        self.testing_phase = testing_phase
        self._optimizer = None
        #self.weights = None

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)
