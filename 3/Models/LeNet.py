from Layers import Initializers, Conv, ReLU, Pooling, SoftMax, FullyConnected, Flatten
from Optimization import Optimizers, Constraints, Loss
import NeuralNetwork
import numpy as np


def build():
    # optimizer parameters
    adam_with_l2 = Optimizers.Adam(5e-4, 0.98, 0.999)
    adam_with_l2.add_regularizer(Constraints.L2_Regularizer(4e-4))

    # layer parameters
    input_image_shape = (1, 28, 28)
    conv_stride_shape = (1, 1)
    convolution_shape = (5, 5)
    c1_kernels, c3_kernels, c5_kernels = 6, 16, 120  # c1_out = 28x28, c3_out = 14x14, c5_out = 7x7 ???
    s2_pooling_shape, s2_stride_shape = (2, 2), (2, 2)
    s4_pooling_shape, s4_stride_shape = (2, 2), (2, 2)
    fc6_units = 84
    softmax_units = 10

    # build network
    net = NeuralNetwork.NeuralNetwork(adam_with_l2,
                                      Initializers.He(),
                                      Initializers.Constant(0.1))
    net.loss_layer = Loss.CrossEntropyLoss()

    cl_1 = Conv.Conv(conv_stride_shape, (1, *convolution_shape), c1_kernels)
    net.append_trainable_layer(cl_1)
    net.layers.append(ReLU.ReLU())
    cl_1_output_shape = (c1_kernels, *input_image_shape[1:])

    s_2 = Pooling.Pooling(s2_stride_shape, s2_stride_shape)
    net.layers.append(s_2)
    #net.layers.append(ReLU.ReLU())

    cl_3 = Conv.Conv(conv_stride_shape, (c1_kernels, *convolution_shape), c3_kernels)
    net.append_trainable_layer(cl_3)
    net.layers.append(ReLU.ReLU())
    m, n = cl_1_output_shape[1] // 4, cl_1_output_shape[2] // 4
    cl_3_output_shape = (c3_kernels, *(m, n))

    s_4 = Pooling.Pooling(s4_stride_shape, s4_stride_shape)
    net.layers.append(s_4)
    #net.layers.append(ReLU.ReLU())
    '''
    cl_5 = Conv.Conv(conv_stride_shape, (c3_kernels, *convolution_shape), c5_kernels)
    net.append_trainable_layer(cl_5)
    net.layers.append(ReLU.ReLU())
    m, n = cl_3_output_shape[1] // 2, cl_3_output_shape[2] // 2
    cl_5_output_shape = (c5_kernels, *(m, n))
    
    
    


    # Flatten-layer & FC-layer needed --> Activations get huge, e.g. max_act = 12e11, we need another solution ???
    net.layers.append(Flatten.Flatten())
    '''

    net.layers.append(Flatten.Flatten())
    fc_5b_in = np.prod(cl_3_output_shape)

    fc_5b = FullyConnected.FullyConnected(input_size= fc_5b_in, output_size=120)
    net.append_trainable_layer(fc_5b)
    net.layers.append(ReLU.ReLU())

    fc_6 = FullyConnected.FullyConnected(input_size=120, output_size=fc6_units)
    net.append_trainable_layer(fc_6)
    net.layers.append(ReLU.ReLU())

    fc_7_in = fc6_units
    fc_7 = FullyConnected.FullyConnected(input_size=fc_7_in, output_size=softmax_units)
    net.append_trainable_layer(fc_7)
    
    


    net.layers.append(SoftMax.SoftMax())

    return net
