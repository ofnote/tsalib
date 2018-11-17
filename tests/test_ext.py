import sys
sys.path.append('../')

import numpy as np
from tsalib import dim_vars
from tsalib import view_transform as vt, permute_transform as pt, expand_transform as et
from tsalib import warp


B, T, D, K = dim_vars('Batch(b):20 SeqLength(t):10 EmbeddingDim(d):100 K(k):1')


def test_reshape():
    x: (B,T,D) = np.ones((B, T, D))

    print (f'Testing Reshape:  x ({x.shape}):')

    h = 4
    print (f'Transforming view {(B,T,D)} to {(B,T,h,D//h)} ')
    #new_shape = vt(src=(B,T,D), to=(B,T,h,D//h), in_shape=x.shape)
    #assert new_shape == (B, T, h, D//h)

    x:(B,T,h,D//h) = x.reshape((B, T, h, D//h))
    print (f'After transform, x : {x.shape}\n')

def test_reshape_short():
    print ('\nTesting Reshape (shorthand)')
    x: (B,T,D) = np.ones((B, T, D))
    h = 4
    print (f'reshape from "btd" -> "b,t,4,d//4"')
    x = x.reshape(vt('btd', f'b,t,{h},d//{h}', x.shape))
    print (f'After transform, x : {x.shape}\n')

    print (f'reshape from "b,t,4,k" -> "b*t,4,k"')
    x1 = x.reshape(vt('b,t,4,k', 'b*t,4,k', x.shape))
    print (f'After transform, x : {x1.shape}\n')

    print (f'reshape from "b,t,," -> "b*t,,"')
    x1 = x.reshape(vt('b,t,,', 'b*t,,', x.shape))
    print (f'After transform, x : {x1.shape}\n')



def test_permute():
    x: (B,T,D,K) = np.ones((B, T, D, K))

    print (f'\nPermuting from {(B,T,D,K)} to {(D,T,B,K)}')
    perm_indices = pt(src=(B,T,D,K), to=(D,T,B,K))
    x = x.transpose(perm_indices)
    print ('permutation order:', perm_indices)
    print (f'After transform, x : {x.shape}\n')
    assert x.shape == (D,T,B,K)

def test_permute_short():
    x: (B,T,D,K) = np.ones((B, T, D, K))
    print (f'\nTesting Permute (shorthand): from "btdk"({x.shape}) to "dtbk"')
    x = x.transpose(pt('btdk','dtbk'))
    print (f'After transform, x : {x.shape}\n')

    print (f'\nTesting Permute (shorthand): from "d_b_"({x.shape}) to "b_d_"')
    x = x.transpose(pt('d_b_','b_d_'))
    print (f'After transform, x : {x.shape}\n')

def test_expand():
    x: (B, T, D) = np.ones((B, T, D))
    x: (B, K, T, D) = x[:, None]

    print (f'Expanding {(B,K,T,D)} by {(K, K*5)}')
    expand_shape = et(src=(B,K,T,D), expansions=[(K, K*5)], in_shape=x.shape)
    print (f'expansion shape: {expand_shape}\n')

def test_expand_short():
    x: 'btd' = np.ones((B, T, D))
    x: 'bktd' = x[:, None]
    print (f'Expanding {(B,K,T,D)} by "k->k*5"')
    expand_shape = et(src=(B,K,T,D), expansions='k->k*5', in_shape=x.shape)
    print (f'expansion shape: {expand_shape}\n')

def test_warp():
    x: 'btd' = np.ones((B, T, D))
    #x = warp(x, 'btd -> b,t,4,d//4 -> b*t,4,d//4', 'vv', debug=True)
    #assert(x.shape == (B*T,4,D//4))

    x = warp(x, 'btd -> b,t,4,d//4 -> b*t,4,d//4 -> b,t,4,d//4 -> btd', 'vvvv', debug=False)
    assert(x.shape == (B,T,D))

    import torch
    y: 'btd' = torch.randn(B, T, D)
    y = warp(y, 'btd -> b,t,4,d//4 -> b,4,t,d//4', 'vp', debug=True)
    assert(y.shape == (B,4,T,D//4))

    print ('test_warp: all assertions hold')


if __name__ == '__main__':
    test_warp()
    '''
    test_reshape()
    test_reshape_short()
    test_permute()
    test_permute_short()
    test_expand()
    test_expand_short()
    '''