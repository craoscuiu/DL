import numpy as np
import copy
from Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self._optimizer = None
        self.input_tensor = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.stored_maxima = None

    def forward(self, input_tensor):
        self.input_tensor = copy.deepcopy(input_tensor)
        py, px = self.pooling_shape
        sy, sx = self.stride_shape
        m, n = input_tensor.shape[2:]
        '''
        1. pooling shape == stride shape
        2. overlapping: pooling shape > stride shape
        3. subsampling: pooling shape < stride shape
        '''
        output_tensor = []
        stored_maxima = []
        for batch in input_tensor:
            stored_max_b = []
            out_b = []
            for c in batch:
                stored_max_c = []
                out_c = []
                for i in range(0, m - py + 1, sy):  # sliced_channel = np.array(c[:m-py+1:sy, :n-px+1:sx])
                    max_row = []
                    for j in range(0, n - px + 1, sx):
                        pool = np.array(c[i:i + py, j:j + px])  # slice channel
                        max_i, max_j = np.unravel_index(np.argmax(pool), pool.shape)  # np.unravel_index(np.nanargmax(pool), pool.shape)
                        max_row.append(np.nanmax(pool))  # add max to j-th value in row vector
                        stored_max_c.append([max_i + i, max_j + j])  # add indices of max to channel storage
                    out_c.append(max_row)  # add row-vector to channel output
                stored_max_b.append(stored_max_c)  # add channel storage to batch storage
                out_b.append(out_c)  # add channel output to batch output
            stored_maxima.append(stored_max_b)  # add batch storage to total storage
            output_tensor.append(out_b)
        output_tensor = np.array(output_tensor)
        self.stored_maxima = np.array(stored_maxima)

        return output_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor.squeeze()
        error_shape = self.input_tensor.shape
        error_tensor2 = np.zeros(error_shape)
        b = 0
        for batch in error_tensor:
            c = 0
            for channel in batch:
                max_pos = self.stored_maxima[b, c]
                i = 0
                for val in channel.reshape(-1, 1):
                    error_tensor2[b, c, max_pos[i, 0], max_pos[i, 1]] = error_tensor2[
                                                                            b, c, max_pos[i, 0], max_pos[i, 1]] + val
                    i += 1
                c += 1
            b += 1
        # in cases where stride < kernel_size the error might be routed multiple times to the same location (sum it up)
        return error_tensor2
