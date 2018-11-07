import sys
sys.path.append('../')

#from typing import List, Sequence, TypeVar

from tsalib import dim_var, dim_vars, declare_common_dim_vars


# global declaration of dimension vars
B, D, V, Dh, T, Te, Td, C, Ci, Co = declare_common_dim_vars()

def test_decls():

    print('\n Test declarations ..')
    #local declarations
    B, C, D, H, W = dim_vars('Batch(b):48 Channels(c):3 EmbedDim(d):300 Height(h) Width(w)')
    print(f'B, C, D, H, W = {B}, {C}, {D}, {H}, {W}')

    #strict=False allows overwriting previous declarations
    H, W = dim_vars ('Height(h):256 Width(w):256', strict=False) 
    print(f'H, W = {H}, {W}')

    return H, W

def test_arith(H, W):
    print('\n Test arithmetic ..')

    # Supports arithmetic over a combination of dim vars and other Python variables
    K = W * 2
    h = 4
    print((h, H // h, K, B*2))

def test_cast_int(D, W):
    print('\n Test integer cast ..')

    x = np.zeros((int(D), int(W)))
    print(f'shape of array: ({D},{W}): {x.shape}')

def test_numpy():
    print('\n Test usage with numpy ..')

    import numpy as np
    a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]])
    print(f'original array: {(B,D)}: {a.shape}')

    b: (2, B, D) = np.stack([a, a])
    print(f'after stack: {(2,B,D)}: {b.shape}')

    ax = (2,B,D).index(B)
    c: (2, D) = np.mean(b, axis=ax)
    print(f'after mean along axis {B}={ax}: {(2,D)}: {c.shape}')




def test_pytorch():
    print('\n Test usage with pytorch ..')

    import torch
    a: (B, D) = torch.Tensor([[1., 2.], [3., 4.]])
    print(f'{(B,D)}: {a.size()}')
    b: (2, B, D) = torch.stack([a, a])
    print(f'{(2,B,D)}: {b.size()}')






if __name__ == '__main__':
    import numpy as np
    H, W = test_decls()
    test_arith(H, W)
    test_cast_int(D, W)

    test_numpy()
    print('')
    test_pytorch()
   