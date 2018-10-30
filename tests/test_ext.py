import sys
sys.path.append('../')

import numpy as np
from tsalib import dim_vars
from tsalib import view_transform, permute_transform, expand_transform



if __name__ == '__main__':
    B, T, D, K = dim_vars('Batch SeqLength EmbeddingDim K')
    H = 4

    x: (20,10,100) = np.ones((20, 10, 100))
    print (f'For x ({x.shape}):\n Transforming view {(B,T,D)} to {(B,T,H,D//H)} ')
    new_shape = view_transform(src=(B,T,D), to=(B,T,H,D//H), in_shape=x.shape)
    x:(20,10,4,25) = x.reshape(new_shape)
    print (f'After transform, x : {x.shape}\n')

    print (f'Permuting from {(B,T,D,K)} to {(D,T,B,K)}')
    perm_indices = permute_transform(src=(B,T,D,K), to=(D,T,B,K))
    x = x.transpose(perm_indices)
    print ('permutation order:', perm_indices)
    print (f'After transform, x : {x.shape}\n')

    x: (B, T, D) = np.ones((20, 10, 100))
    x: (B, K, T, D) = x[:, None]
    print (f'Expanding {(B,K,T,D)} by {(K, K*5)}')
    expand_shape = expand_transform(src=(B,K,T,D), expansions=[(K, K*5)], in_shape=x.shape)
    print (f'expansion shape: {expand_shape}\n')
