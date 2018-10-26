import sys
sys.path.append('../')

#from typing import List, Sequence, TypeVar

from tsalib import TS, declare_common_dim_vars, decl_dim_vars

# definitions in tsalib/ts.py
B, D, V, Dh, T, Te, Td, C, Ci, Co = declare_common_dim_vars()
H, W = decl_dim_vars ('Height Width')


def test_numpy():
    import numpy as np
    a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]])
    print(f'original array: {(B,D)}: {a.shape}')
    
    b: (2, B, D) = np.stack([a, a])
    print(f'after stack: {(2,B,D)}: {b.shape}')

    ax = (2,B,D).index(B)
    c: (2, D) = np.mean(b, axis=ax)
    print(f'after mean along axis {B}={ax}: {(2,D)}: {c.shape}')

    # Supports arithmetic over a combination of dim vars and other Python variables
    K = W * 2
    var1 = 10
    print((...,4, H // 4, K, var1))


def test_pytorch():

    import torch
    a: (B, D) = torch.Tensor([[1., 2.], [3., 4.]])
    print(f'{(B,D)}: {a.size()}')
    b: (2, B, D) = torch.stack([a, a])
    print(f'{(2,B,D)}: {b.size()}')


    # Supports arithmetic over dim vars and other Python variables
    K = W * 2
    var1 = 10
    print((4, H // 4, K, var1))




if __name__ == '__main__':
    test_numpy()
    test_pytorch()
   