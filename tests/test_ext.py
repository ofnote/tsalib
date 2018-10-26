import sys
sys.path.append('../')

import numpy as np
from tsalib.ext import view_transform, permute_transform



if __name__ == '__main__':
    from tsalib.ts import decl_dim_vars
    B, T, D = decl_dim_vars('Batch SeqLength EmbeddingDim')
    H = 5

    x = np.ones((20, 10, 100))
    print (f'For x ({x.shape}):\n Transforming view {(B,T,D)} to {(B,T,H,D//H)} ')
    new_shape = view_transform(src=(B,T,D), to=(B,T,H,D//H), in_shape=x.shape)
    x = x.reshape(new_shape)
    print (f'After transform, x : {x.shape}')

    print (f'Permuting from {(B,T,D)} to {(D,T,B)}')
    res = permute_transform(src=(B,T,D), to=(D,T,B))
    print ('permutation order:', res)