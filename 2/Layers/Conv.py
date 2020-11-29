import numpy as np
from Layers import Initializers
from scipy import ndimage, signal

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
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None

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
        batch = error_tensor.shape[0]
        total_weights = self.weights
        weights_shape = (self.input_tensor.shape[1], self.num_kernels ,) + self.convolution_shape[1:]
        back_weights = np.zeros(weights_shape)
        output_tensor = np.zeros(self.input_tensor.shape)

        for m in range(self.input_tensor.shape[1]):
            for n in range(self.num_kernels):
                if n == 0:
                    out = total_weights[n][m]
                else:
                    out = np.dstack((out, total_weights[n][m]))

            out = np.reshape(out, back_weights[m].shape)
            out = np.flip(out, axis = 1 )

            back_weights[m] = out


        for i in range(batch):
            input = error_tensor[i]
            print("CAsor ", input.shape)
            output = output_tensor[i]
            for n in range(self.input_tensor.shape[1]):
                weights = back_weights[n]
                if self.bias.size == 1:
                    bias = self.bias
                else:
                    bias = self.bias[n]

                input = input - bias

                if len(weights.shape) == 2 and len(input.shape) == 2:
                    z,y = input.shape
                    upsampled_input = np.zeros((z,self.input_tensor.shape[2]))
                    upsampled_input[::,::self.stride_shape[0]] = input
                    result = 0
                    for c in range(self.num_kernels):
                        result = result + ndimage.convolve1d(upsampled_input[c], weights[c])
                else:
                    z,y,x = input.shape
                    result = np.zeros((z, self.input_tensor.shape[2],self.input_tensor.shape[3] ))
                    result[::, ::self.stride_shape[0], ::self.stride_shape[1]] = input
                    row_stride = ((result.shape[1] - 1) - result.shape[1] + weights.shape[1]) / 2
                    column_stride = ((result.shape[2] - 1) - result.shape[2] + weights.shape[2]) / 2
                    result = np.pad(result, ((0, 0),(int(np.ceil(row_stride)), int(np.floor(row_stride))),
                                                         (int(np.ceil(column_stride)), int(np.floor(column_stride)))),
                                          mode='constant', constant_values=0)
                    result = signal.convolve(result,weights, mode = 'valid')

                output[n] = result
                #stride subsampling


            output_tensor[i] = output

        return output_tensor


    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_tensor):
        self._gradient_weights = gradient_tensor

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_weights(self, gradient_tensor):
        self._gradient_bias = gradient_tensor