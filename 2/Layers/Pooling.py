import numpy as np
import copy


class Pooling:

    def __init__(self, stride_shape, pooling_shape):
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
                        max_i, max_j = np.unravel_index(np.nanargmax(pool), pool.shape)
                        max_row.append(np.nanmax(pool))  # add max to j-th value in row vector
                        stored_max_c.append([max_i + i, max_j + j])  # add indices of max to channel storage
                    out_c.append(max_row)  # add row-vector to channel output
                stored_max_b.append(stored_max_c)  # add channel storage to batch storage
                out_b.append(out_c)  # add channel output to batch output
            stored_maxima.append(stored_max_b)  # add batch storage to total storage
            output_tensor.append(out_b)
        output_tensor = np.array(output_tensor)
        self.stored_maxima = np.array(stored_maxima)
        '''
        # Case 1 - forward output tensor
        ny = m//py  # num of pads in y-direction
        nx = n//px  # num of pads in x-direction
        valid_pool = input_tensor[..., :ny*py, :nx*px]
        pool_shape = input_tensor.shape[:2] + (ny, py, nx, px)
        valid_pool = valid_pool.reshape(pool_shape)
        output_tensor = np.nanmax(valid_pool, axis=(3, 5))
        # Case 1 - index locations
        index_shape = input_tensor.shape[:2] + (ny*py, nx*px)
        index_pool = valid_pool.reshape(index_shape)
        ind_m, ind_n = np.unravel_index(index_pool.nanargmax(0), (py, px))
        '''

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
                    error_tensor2[b, c, max_pos[i, 0], max_pos[i, 1]] = error_tensor2[b, c, max_pos[i, 0], max_pos[i, 1]] + val
                    i += 1
                c += 1
            b += 1
        # in cases where stride < kernel_size the error might be routed multiple times to the same location (sum it up)
        return error_tensor2


def main(self):
    mat = np.array([[20, 200, -5, 23, 7],
                    [-13, 134, 119, 100, 8],
                    [120, 32, 49, 25, 12],
                    [-120, 12, 9, 23, 15],
                    [-57, 84, 19, 17, 82],
                    ])
    # soln
    # [200, 119, 8]
    # [120, 49, 15]
    # [84, 19, 82]
    M, N = mat.shape
    K = 2
    L = 2

    MK = M // K
    NL = N // L

    # split the matrix into 'quadrants'
    Q1 = mat[:MK * K, :NL * L].reshape(MK, K, NL, L)
