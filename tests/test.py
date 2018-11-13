import sys
sys.path.append('../')

#from typing import List, Sequence, TypeVar

from tsalib import dim_var, dim_vars, declare_common_dim_vars


# global declaration of dimension vars
#B, D, V, Dh, T, Te, Td, C, Ci, Co = declare_common_dim_vars()
B, C, D, H, W = dim_vars('Batch(b):48 Channels(c):3 EmbedDim(d):300 Height(h) Width(w)')

def test_decls():

    print('\n Test declarations ..')
    #local declarations
    print(f'B, C, D = {B}, {C}, {D}')

    #strict=False allows overwriting previous declarations
    H, W = dim_vars ('Height(h):256 Width(w):256', strict=False) 
    print(f'H, W = {H}, {W}')


def test_arith():
    print('\n Test arithmetic ..')

    # Supports arithmetic over a combination of dim vars and other Python variables
    K = W * 2
    h = 4
    print((h, H // h, K, B*2))

def test_cast_int():
    print('\n Test integer cast ..')

    x = np.zeros((B, C))
    print(f'shape of array: ({B},{C}): {x.shape}')

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
    B, D = dim_vars('Batch:2 EmbedDim:3')
    import torch

    a = torch.Tensor([[1., 2., 4.], [3., 6., 9.]])
    assert a.size() == (B, D)

    b = torch.stack([a, a])

    print ('Asserting b.size() == (2,B,D)')
    assert b.size() == (2, B, D)

    c = torch.cat([a, a], dim=1)
    assert c.size() == (B, D*2)








if __name__ == '__main__':
    import numpy as np
    test_decls()
    test_arith()
    test_cast_int()
    test_numpy()
    test_pytorch()
   