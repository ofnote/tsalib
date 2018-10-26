# Tensor Shape Annotation Library (tsalib)

Writing programs which manipulate tensors (e.g., using `numpy`, `pytorch`, `tensorflow`, ..) requires you to carefully keep track of shapes of tensor variables. Carrying around the shapes in your head gets increasingly hard as programs become more complex, e.g., when creating a new `RNN` cell or designing a new kind of `attention` mechanism or trying to do a surgery of non-trivial pre-trained architectures (`resnet101`, `densenet`). There is no principled way of shape tracking inside code -- most developers resort to writing adhoc comments embedded in code to keep track of tensor shapes.

The `tsalib` library comes to your rescue here. It allows you to label tensor variables with their shapes directly in the code, as *first-class* type annotations. Shape annotations turn out to be useful in many ways. They help you quickly cross-check the variable shapes when writing new transformations or modifying existing modules. Moreover, the annotations serve as useful documentation to guide others trying to understand or extend your module.

* Because shapes can be dynamic, you can annotate with `symbolic` shape expressions over named dimension variables, e.g., 

    ```python
    v: (B, C, H, W) = torch.randn(batch_size, channels, h, w)
    v: (B, C, H//2, W//2) = maxpool(v)

    ```

    Here `B`, `C`, `H`, `W` are pre-defined named dimension variables. It is easy to define new named dimensions customized to your network architecture. Of course, use constant values if one or more dimensions are always fixed.

    `v : (B, 64, H, W) = torch.randn(batch_size, 64, h, w)`


* Works seamlessly with arbitrary tensor libraries:  `numpy`, `pytorch`, `tensorflow`, `mxnet`, etc. 

* Faster debugging: if you annotate-as-you-go, the tensor variable shapes are always explicit in code. No more adhoc shape `print`s when debugging shape errors. 

## Getting Started

See [tests/test.py](tests/test.py) to get started quickly.

```python
from tsalib import TS, decl_dim_vars
import numpy as np

#declare named dimension variables
B, C, H, W = TS('Batch'), TS('Channels'), TS('Height'), TS('Width')
#or
B, C, H, W = decl_dim_vars('Batch Channels Height Width')

#now build expressions over dimension variables and annotate tensor variables

a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]])
print(f'original array: {(B,D)}: {a.shape}') #(Batch, EmbedDim): (2, 3)

b: (2, B, D) = np.stack([a, a])
print(f'after stack: {(2,B,D)}: {b.shape}') #(2, Batch, EmbedDim): (2, 2, 3)

ax = (2,B,D).index(B) #ax = 1
c: (2, D) = np.mean(b, axis=ax) 
print(f'after mean along axis {B}={ax}: {(2,D)}: {c.shape}') #... axis Batch=1: (2, EmbedDim): (2, 3)
```

Arithmetic over shapes is supported:

```python
v: (B, C, H, W) = torch.randn(batch_size, channels, h, w)
x : (B, C * 2, H//2, W//2) = torch.nn.conv2D(ch_in, ch_in*2, ...)(v) 
```

Shapes can be manipulated like ordinary `tuples`:

```python 
S = (B, C*2, H, W)
print (S[:-2]) #(Batch, 2*Channels)
```

 The [examples](examples) directory contains complex multi-module networks annotated with TSAs: [resnet](examples/resnet.py), [transformer](examples/openai_transformer.py)

## Dependencies

Python >= 3.6. Allows optional type annotations for variables. These annotations do not affect the program performance in any way. 

`sympy`. A library for building symbolic expressions in Python.

## Installation

`pip install tsalib`

## References

* A [proposal](https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY/edit#heading=h.rkj7d39awayl) for designing a tensor library with named dimensions from ground-up. The TSA library takes care of some use cases, without requiring any change in the tensor libraries.
* Pytorch Issue on Names Axes [here](https://github.com/pytorch/pytorch/issues/4164).

## Contributors

Nishant Sinha, OffNote Labs


