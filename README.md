# Tensor Shape Annotation Library (tsalib)

The Tensor Shape Annotation (TSA) library enables first-class, embedded annotations of tensor variables. These annotations improve developer productivity, accelerate debugging and enhance code readability. Detailed blog article [here](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b).

## Introduction

Writing deep learning programs which manipulate tensors (e.g., using `numpy`, `pytorch`, `keras`, `tensorflow`, ..) requires you to carefully keep track of shapes of tensor variables. Carrying around the shapes in your head gets increasingly hard as programs become more complex, e.g., when creating a new `RNN` cell or designing a new kind of `attention` mechanism or trying to do a surgery of non-trivial pre-trained architectures (`resnet101`, `densenet`). There is no principled way of shape tracking inside code -- most developers resort to writing adhoc comments embedded in code to keep track of tensor shapes.

`tsalib` comes to our rescue here. It allows us to label tensor variables with their shapes directly in the code, as *first-class* type annotations. Shape annotations turn out to be useful in many ways. They help us to quickly cross-check the variable shapes when *debugging* or writing new transformations or modifying existing modules. Moreover, the annotations serve as useful documentation to help others in understanding or extending your module.

* Because shapes can be dynamic, annotate tensors with `symbolic` shape expressions over named dimension variables, with arithmetic:

    ```python
    v: (B, C, H, W) = torch.randn(batch_size, channels, h, w)
    v: (B, C, H // 2, W // 2) = maxpool(v)

    ```

    Here `B`, `C`, `H`, `W` are pre-defined named dimension variables. It is easy to define new named dimensions customized to your network architecture. Of course, use constant values if one or more dimensions are always fixed.

    `v : (B, 64, H, W) = torch.randn(batch_size, 64, h, w)`


* Works seamlessly with arbitrary tensor libraries:  `numpy`, `pytorch`, `keras`, `tensorflow`, `mxnet`, etc. Use TSAs to improve code clarity everywhere, even in your machine learning data pipelines.

* Faster debugging: if you annotate-as-you-go, the tensor variable shapes are explicit in code, readily available for a quick inspection. No more adhoc shape `print`ing when investigating obscure shape errors. 

## Getting Started

See [tests/test.py](tests/test.py) to get started quickly.

```python
from tsalib import dim_var as dv, dim_vars as dvs
import numpy as np

#declare named dimension variables
B, C, H, W = dv('Batch'), dv('Channels'), dv('Height'), dv('Width')
#or
B, C, H, W = dvs('Batch Channels Height Width')

#now build expressions over dimension variables and annotate tensor variables

a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]])
print(f'original array: {(B,D)}: {a.shape}') #(Batch, EmbedDim): (2, 3)

b: (2, B, D) = np.stack([a, a])
print(f'after stack: {(2,B,D)}: {b.shape}') #(2, Batch, EmbedDim): (2, 2, 3)

#use dim vars to write better code

ax = (2, B, D).index(B) #ax = 1
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

## Examples

 The [examples](examples) directory contains TS annotations of a few well-known, complex neural architectures: [resnet](examples/resnet.py), [transformer](examples/openai_transformer.py). With TSAs, the `forward` function gives a deeper insight into how the module works. 

## Dependencies

Python >= 3.6. Allows optional type annotations for variables. These annotations do not affect the program performance in any way. 

`sympy`. A library for building symbolic expressions in Python.

## Installation

`pip install tsalib`

## Going further
Once we have named dimensions in the code, we can exploit them to further improve code productivity and clarity.

* Avoid explicit shape computations for `reshaping`. Use `tsalib.view_transform` to specify view changes declaratively.

```python
    x = np.ones((20, 10, 300))
    new_shape = view_transform(src=(B,T,D), to=(B,T,4,D//4), in_shape=x.shape)
    x = x.reshape(new_shape) #(20, 10, 300) -> (20, 10, 4, 75)
```

* Similarly, use `tsalib.permute_transform` to compute permutation index order from a declarative spec. 
```python 
    perm_indices = permute_transform(src=(B,T,D), to=(D,T,B))
    x = x.transpose(perm_indices) #(10, 50, 300) -> (300, 50, 10)
```

See [tests/test_ext.py](tests/test_ext.py) for complete examples.

## References

* A [proposal](https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY/edit#heading=h.rkj7d39awayl) for designing a tensor library with named dimensions from ground-up. The TSA library takes care of some use cases, without requiring any change in the tensor libraries.
* Pytorch Issue on Names Axes [here](https://github.com/pytorch/pytorch/issues/4164).
* Using [einsum](http://ajcr.net/Basic-guide-to-einsum/) for tensor operations improves productivity and code readability. [blog](https://rockt.github.io/2018/04/30/einsum)
* The [Tile](https://vertexai-plaidml.readthedocs-hosted.com/en/latest/writing_tile_code.html) DSL uses indices ranging over dimension variables to write compact, library-independent tensor operations.

## Contributors

Nishant Sinha, OffNote Labs. @[medium](https://medium.com/@ekshakhs), @[twitter](https://twitter.com/ekshakhs)


