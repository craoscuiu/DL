import numpy as np
from Layers import Initializers
from Optimization import Optimizers
from scipy import ndimage, signal
from copy import deepcopy

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels ):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # initialize n amount of fkernels
        shape_weights = (num_kernels,) + convolution_shape
        fan_in = np.prod(convolution_shape)
        fan_out = np.prod(convolution_shape[1:]) * num_kernels
        self.weights = Initializers.UniformRandom().initialize(shape_weights, fan_in, fan_out)
        self.bias = Initializers.Constant(np.random.uniform(0,1,1)).initialize(num_kernels,fan_in, fan_out)

        self.input_tensor =  0
        self.optimizer = Optimizers.Sgd(0)
        self._gradient_weights = np.random.uniform(0,10,shape_weights)
        self._gradient_bias = np.random.uniform(0,10,num_kernels)
        self._weights_optimizer = deepcopy(self.optimizer)
        self._bias_optimizer = deepcopy(self.optimizer)


    def initialize(self, weights_initializer, bias_initializer):
        shape_weights = (self.num_kernels,) + self.convolution_shape
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(shape_weights, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.num_kernels, fan_in, fan_out)


    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        batch = input_tensor.shape[0]

        #determining output shape
        total_weights = self.weights
        output_row = np.ceil(input_tensor.shape[2] / self.stride_shape[0])
        row_stride = ((input_tensor.shape[2] - 1) - input_tensor.shape[2] + total_weights.shape[2]) / 2


        if len(input_tensor.shape) == 4:
            output_column = np.ceil(input_tensor.shape[3] / self.stride_shape[1])
            column_stride = ((input_tensor.shape[3] - 1) - input_tensor.shape[3] + total_weights.shape[3]) / 2
            input_tensor = np.pad(input_tensor, ((0, 0), (0, 0),
                                                 (int(np.ceil(row_stride)), int(np.floor(row_stride))),
                                                 (int(np.ceil(column_stride)), int(np.floor(column_stride)))),
                                  mode='constant', constant_values=0)
            output_tensor = np.zeros((batch, self.num_kernels, int(output_row), int(output_column)))
        else:
            output_tensor = np.zeros((batch, self.num_kernels, int(output_row)))

        for i in range(batch):
            input = input_tensor[i]
            output = output_tensor[i]
            for n in range(self.num_kernels):
                weights = total_weights[n]

                if len(weights.shape) == 2 and len(input.shape) == 2:
                    test = 0
                    for c in range(3):
                        test = test + ndimage.correlate1d(input[c], weights[c])
                    test = test[::self.stride_shape[0]]


                else:
                    test = signal.correlate(input,weights, mode = 'valid')
                    test = np.squeeze(test, axis=(0))
                    test = test[::self.stride_shape[0], ::self.stride_shape[1]]

                #stride subsampling

                if self.bias.size == 1:
                    bias = self.bias
                else:
                    bias = self.bias[n]

                output[n] = test + bias
            output_tensor[i] = output

        return output_tensor

    def backward(self,error_tensor):
        self._weights_optimizer = deepcopy(self.optimizer)
        self._bias_optimizer = deepcopy(self.optimizer)

        batch = error_tensor.shape[0]
        total_weights = self.weights
        weights_shape = (self.input_tensor.shape[1], self.num_kernels ,) + self.convolution_shape[1:]
        back_weights = np.zeros(weights_shape)
        output_tensor = np.zeros(self.input_tensor.shape)

        for m in range(self.input_tensor.shape[1]):
            for n in range(self.num_kernels):
                back_weights[m][n] = total_weights[self.num_kernels - 1 - n ][m]
            n = 0

        gradient_tensor = np.zeros(self._gradient_weights.shape)
        bias_tensor = np.zeros(self._gradient_bias.shape)

        for i in range(batch):
            error = error_tensor[i]
            output = output_tensor[i]

            for n in range(self.input_tensor.shape[1]):
                weights = back_weights[n]
                if self.bias.size == 1:
                    bias = self.bias
                else:
                    bias = self.bias[n]

                if len(weights.shape) == 2 and len(error.shape) == 2:
                    z,y = error.shape
                    upsampled_error = np.zeros((z,self.input_tensor.shape[2]))
                    upsampled_error[::,::self.stride_shape[0]] = error
                    result = 0
                    for c in range(self.num_kernels):
                        result = result + ndimage.convolve1d(upsampled_error[c], weights[c])
                else:
                    z,y,x = error.shape
                    upsampled_error = np.zeros((z, self.input_tensor.shape[2],self.input_tensor.shape[3] ))
                    upsampled_error[::, ::self.stride_shape[0], ::self.stride_shape[1]] = error
                    row_stride = ((upsampled_error.shape[1] - 1) - upsampled_error.shape[1] + weights.shape[1]) / 2
                    column_stride = ((upsampled_error.shape[2] - 1) - upsampled_error.shape[2] + weights.shape[2]) / 2
                    padded_error = np.pad(upsampled_error, ((0, 0),(int(np.ceil(row_stride)), int(np.floor(row_stride))),
                                                         (int(np.ceil(column_stride)), int(np.floor(column_stride)))),
                                          mode='constant', constant_values=0)
                    result = signal.convolve(padded_error,weights, mode ='valid')

                output[n] = result
                #stride subsampling


            input = self.input_tensor[i]
            pad_length = ( self.convolution_shape[1] - 1 )/2
            if len(weights.shape) != 2 and len(error.shape) != 2:
                pad_width = ( self.convolution_shape[2] - 1 )/2

            for k in range(self.num_kernels):
                error_new = np.expand_dims(upsampled_error[k], axis = 0 )
                if len(weights.shape) != 2 and len(error.shape) != 2:
                   padded_input = np.pad(input, ((0, 0),(int(np.ceil(pad_length)),int(np.floor(pad_length))),
                                              (int(np.ceil(pad_width)),int(np.floor(pad_width))))
                                      , mode ='constant', constant_values=0)
                else:
                    padded_input = np.pad(input, ((0, 0), (int(np.ceil(pad_length)), int(np.floor(pad_length))))
                                          , mode='constant', constant_values=0)
                test = signal.correlate(error_new, padded_input, mode='valid')
                gradient_tensor[k] = gradient_tensor[k] + signal.correlate(padded_input,error_new ,mode = 'valid')
                bias_tensor[k] = bias_tensor[k] + np.sum(error[k])


            
            output_tensor[i] = output

        self.weights = self._weights_optimizer.calculate_update(self.weights, gradient_tensor)
        self.bias = self._bias_optimizer.calculate_update(self.bias, bias_tensor)
        self._gradient_weights = deepcopy(gradient_tensor)
        self._gradient_bias = deepcopy(bias_tensor)

        return output_tensor



    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    @property
    def weights_optimizer(self):
        return self._weights_optimizer

    @weights_optimizer.setter
    def weights_optimizer(self, w_optimizer):
        self._weights_optimizer = w_optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, b_optimizer):
        self._bias_optimizer = b_optimizer