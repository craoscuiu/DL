import numpy as np
from Layers import Initializers
from Optimization import Optimizers
from scipy import ndimage, signal
from copy import deepcopy


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # initialize n amount of fkernels
        self.shape_weights = (num_kernels,) + convolution_shape
        self.fan_in = np.prod(convolution_shape)
        self.fan_out = np.prod(convolution_shape[1:]) * num_kernels
        # default initializers and optimizers
        self.weights = Initializers.UniformRandom().initialize(self.shape_weights, self.fan_in, self.fan_out)
        self.bias = Initializers.Constant(np.random.uniform(0, 1, 1)).initialize(num_kernels, self.fan_in, self.fan_out)

        self.input_tensor = 0
        self._optimizer = Optimizers.Sgd(0)
        self._gradient_weights = np.random.uniform(0, 1, self.shape_weights)
        self._gradient_bias = np.random.uniform(0, 1, num_kernels)
        self.weights_optimizer = None
        self.bias_optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.shape_weights, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.num_kernels, self.fan_in, self.fan_out)

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        batch = input_tensor.shape[0]

        # determining output shape and the necessary operators padding operations
        total_weights = self.weights
        output_row = np.ceil(input_tensor.shape[2] / self.stride_shape[0])
        row_pad = ((input_tensor.shape[2] - 1) - input_tensor.shape[2] + total_weights.shape[2]) / 2

        # padding the input tensors and determining output tensor shape, separate case whether it's 1D or 3D case
        if len(input_tensor.shape) == 4:
            output_column = np.ceil(input_tensor.shape[3] / self.stride_shape[1])
            column_pad = ((input_tensor.shape[3] - 1) - input_tensor.shape[3] + total_weights.shape[3]) / 2
            input_tensor = np.pad(input_tensor, ((0, 0), (0, 0),
                                                 (int(np.ceil(row_pad)), int(np.floor(row_pad))),
                                                 (int(np.ceil(column_pad)), int(np.floor(column_pad)))),
                                  mode='constant', constant_values=0)
            output_tensor = np.zeros((batch, self.num_kernels, int(output_row), int(output_column)))
        else:
            output_tensor = np.zeros((batch, self.num_kernels, int(output_row)))

        # loop over batch
        for i in range(batch):
            input = input_tensor[i]

            # loop over the number of kernels
            for n in range(self.num_kernels):
                weights = total_weights[n]

                # do the correlation depending on whether it's 1D ( correlation over the channels ) or 3D ( valid convolution )
                if len(weights.shape) == 2 and len(input.shape) == 2:
                    test = 0
                    for c in range(3):
                        test = test + ndimage.correlate1d(input[c], weights[c])
                    # downsampling to simulate strided convolution
                    test = test[::self.stride_shape[0]]
                else:
                    test = signal.correlate(input, weights, mode='valid')
                    test = np.squeeze(test, axis=(0))

                    test = test[::self.stride_shape[0], ::self.stride_shape[1]]
                #adding bias
                bias = self.bias[n]
                output_tensor[i][n] = test + bias

        return output_tensor

    def backward(self, error_tensor):
        #copying optimizers for weights and bias update
        if self.weights_optimizer == None and self.bias_optimizer == None:
            self.weights_optimizer = deepcopy(self._optimizer)
            self.bias_optimizer = deepcopy(self._optimizer)

        #initializing tensors and parameters used for backwards method
        gradient_tensor = np.zeros(self._gradient_weights.shape)
        bias_tensor = np.zeros(self._gradient_bias.shape)
        batch = error_tensor.shape[0]
        total_weights = self.weights
        output_tensor = np.zeros(self.input_tensor.shape)

        # weights swap ( so now it has the shape of channel depth, num kernels, x, y ) and flip operation
        axis_array = np.arange(len(total_weights.shape)) #0,1, 2, 3
        axis_array[0], axis_array[1] = axis_array[1], axis_array[0] # 1,0,2,3
        back_weights = np.flip(np.transpose(total_weights, axis_array), axis=1)

        # upsampling of error tensor and then padding it before the convolution operation
        if len(total_weights.shape) == 4:
                # upsampling of error tensor
                b, z, y, x = error_tensor.shape
                upsampled_error = np.zeros((b, z, self.input_tensor.shape[2], self.input_tensor.shape[3]))
                upsampled_error[..., ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

                # padding of error tensor for weights update operation
                row_pad = ((upsampled_error.shape[2] - 1) - upsampled_error.shape[2] + back_weights.shape[2]) / 2
                column_pad = ((upsampled_error.shape[3] - 1) - upsampled_error.shape[3] + back_weights.shape[3]) / 2
                padded_error = np.pad(upsampled_error,
                                      ((0,0),(0, 0), (int(np.ceil(row_pad)), int(np.floor(row_pad))),
                                       (int(np.ceil(column_pad)), int(np.floor(column_pad)))),
                                      mode='constant', constant_values=0)
        else:
            # upsample 1D error tensor basing on the stride shapes
            b, z, y = error_tensor.shape
            upsampled_error = np.zeros((b, z, self.input_tensor.shape[2]))
            upsampled_error[..., ::self.stride_shape[0]] = error_tensor

        # looping over batch
        for i in range(batch):
            error = error_tensor[i]

            # looping over every layer of the channel
            for n in range(self.input_tensor.shape[1]):
                # picking one of the modified weight kernels for backwards convolution
                weights = back_weights[n]

                # main backward convolution operation, separate case between 1D ( loop over layer of error tensor ) and 3D
                if len(weights.shape) == 2 and len(error.shape) == 2:
                    result = 0
                    for c in range(self.num_kernels):
                        result = result + ndimage.convolve1d(upsampled_error[i][c], weights[c])
                else:
                    result = signal.convolve(padded_error[i], weights, mode='valid')

                # gives out result of backward convolution , which is gradient over the previous layer
                output_tensor[i][n] = result

            # initialization parameters for gradient over weights and biases
            input = self.input_tensor[i]
            pad_length = (self.convolution_shape[1] - 1) / 2

            # padding the input tensor, depending on whether it's 1D or 3D case
            if len(weights.shape) != 2 and len(error.shape) != 2:
                pad_width = (self.convolution_shape[2] - 1) / 2
                padded_input = np.pad(input, ((0, 0), (int(np.ceil(pad_length)), int(np.floor(pad_length))),
                                              (int(np.ceil(pad_width)), int(np.floor(pad_width))))
                                      , mode='constant', constant_values=0)
            else:
                padded_input = np.pad(input, ((0, 0), (int(np.ceil(pad_length)), int(np.floor(pad_length))))
                                      , mode='constant', constant_values=0)

            #determine gradient over weights and biases, this is done by looping over the layers of the error tensor
            for k in range(self.num_kernels):
                error_new = np.expand_dims(upsampled_error[i][k], axis=0)
                gradient_tensor[k] = gradient_tensor[k] + signal.correlate(padded_input, error_new, mode='valid')
                bias_tensor[k] = bias_tensor[k] + np.sum(error[k])

        #update weights, biases and the gradient properties
        self._gradient_weights = deepcopy(gradient_tensor)
        self._gradient_bias = deepcopy(bias_tensor)
        self.weights = self.weights_optimizer.calculate_update(self.weights, self._gradient_weights)
        self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)


        return output_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, i_gradient_weights):
        self._gradient_weights = i_gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, i_gradient_bias):
        self._gradient_bias = i_gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, i_optimizer):
        self._optimizer = i_optimizer
