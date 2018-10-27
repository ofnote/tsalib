import sys
sys.path.append('../')

import numpy as np
from tsalib.ext import view_transform, permute_transform



if __name__ == '__main__':
    from tsalib.ts import decl_dim_vars
    B, T, D, K = decl_dim_vars('Batch SeqLength EmbeddingDim K')
    H = 4

    x = np.ones((20, 10, 100))
    print (f'For x ({x.shape}):\n Transforming view {(B,T,D)} to {(B,T,H,D//H)} ')
    new_shape = view_transform(src=(B,T,D), to=(B,T,H,D//H), in_shape=x.shape)
    x = x.reshape(new_shape)
    print (f'After transform, x : {x.shape}')

    print (f'Permuting from {(B,T,D,K)} to {(D,T,B,K)}')
    perm_indices = permute_transform(src=(B,T,D,K), to=(D,T,B,K))
    x = x.transpose(perm_indices)
    print ('permutation order:', perm_indices)
    print (f'After transform, x : {x.shape}')
