# Tensor Shape Annotation Library (tsalib)

The Tensor Shape Annotation (TSA) library enables you to write first-class, library-independent, **shape expressions** over **named dimensions** for matrix/tensor variables.
They can be used to annotate variables *inline* with their shapes in your deep learning/tensor program.
They also enable us to write more *fluent* shape transformations and matrix/tensor operations. Using TSAs enhances code clarity, accelerates debugging and improves overall developer productivity when writing tensor programs. 
Detailed article [here](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b).

## Introduction

Writing deep learning programs which manipulate multi-dimensional tensors (with `numpy`, `pytorch`, `keras`, `tensorflow`, ..) requires you to carefully keep track of shapes of tensor variables. Carrying around the shapes in your head gets increasingly hard as programs become more complex, e.g., reshaping before a `matmult`, doing a surgery of deep pre-trained architectures (`resnet101`, `densenet`), designing a new kind of `attention` mechanism or when creating a new `RNN` cell. There is no principled way of shape specification and tracking inside code -- most developers resort to writing adhoc comments embedded in code to keep track of tensor shapes.

`tsalib` comes to our rescue here. It allows you to write shape *expressions* describing the shape of tensor variables. These expressions can be used in multiple ways, e.g., as *first-class* type annotations of variables, or to specify shape transformations (`reshape`, `permute`, `expand`) or a tensor product operations (`matmult`) succinctly. TSAs expose the typically *invisible* tensor shape types, leading to improved productivity across the board. 

## Dimension Variables

Tensor shape annotations (TSAs) are constructed using `dimension` variables: `B` (Batch), `C` (Channels), `D` (EmbedDim), Integer constants, and arithmetic expressions (`B*2`, `C+D`) over them. 

TSAs may be be represented as
* a tuple `(B,H,D)` [long form]
* a string `'b,h,d'` (compact notation) (or simply `'bhd'`)

Here is an example snippet which uses TSAs in a `pytorch` program to describe how program operations change the input shape incrementally. TSAs work seamlessly with arbitrary tensor libraries:  `numpy`, `pytorch`, `keras`, `tensorflow`, `mxnet`, etc.

```python
from tsalib import dim_vars as dvs
B, C, H, W = dvs('Batch Channels Height Width') #declare dimension variables
...
v: (B, C, H, W) = torch.randn(batch_size, n_channels, h, w) #create tensor
v: (B, C, H // 2, W // 2) = maxpool(v) #torch maxpool operation

#or, if n_channels is fixed:
v : (B, 64, H, W) = torch.randn(batch_size, 64, h, w)

``` 

Shape annotations turn out to be useful in many ways. 
* They help us to quickly cross-check the variable shapes when writing new transformations or modifying existing modules.
* Faster *debugging*: if you annotate-as-you-go, the tensor variable shapes are explicit in code, readily available for a quick inspection. No more adhoc shape `print`ing when investigating obscure shape errors. 
* Use TSAs to improve code clarity everywhere, even in your machine learning data pipelines.
* They serve as useful documentation to help others understand or extend your module.


## Installation

`pip install tsalib`

## Getting Started

See [tests/test.py](tests/test.py) for complete examples.

```python
from tsalib import dim_var as dv, dim_vars as dvs
import numpy as np
```

### Declare Dimension Variables, Expressions over them
```python
B, C, D, H, W = dv('Batch'), dv('Channels'), dv('EmbedDim'), dv('Height'), dv('Width')
#or
B, C, D, H, W = dvs('Batch Channels EmbedDim Height Width')
#or declare dim vars with optional integer value
B, C, D, H, W = dvs('Batch:48 Channels:3 EmbedDim:300 Height Width')
#or provide *shorthand* names for dim vars
B, C, D, H, W = dvs('Batch(b):48 Channels(c):3 EmbedDim(d):300 Height(h) Width(w)')

S1 = (B, C, D)
S2 = (B*C, D//2)
```


### Use TSAs to annotate variables on-the-go (Python 3)

```python
a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]]) #(Batch, EmbedDim): (2, 3)

b: (2, B, D) = np.stack([a, a]) #(2, Batch, EmbedDim): (2, 2, 3)
```

Arithmetic over dimension variables is supported. This enables easy tracking of shape changes across neural network layers.

```python
v: (B, C, H, W) = torch.randn(batch_size, channels, h, w)
x : (B, C * 2, H//2, W//2) = torch.nn.conv2D(ch_in, ch_in*2, ...)(v) 
```

### Use TSAs to make matrix operations compact and explicit


Avoid explicit shape computations for `reshaping`. Use `tsalib.view_transform` to specify view changes declaratively.

```python
    x: (20,10,300) = np.ones((20, 10, 300))
    new_shape = view_transform(src=(B,T,D), to=(B,T,4,D//4), in_shape=x.shape)
    x = x.reshape(new_shape) #(20, 10, 300) -> (20, 10, 4, 75)
   
    #or, compact form:
    x = x.reshape(vt('btd', 'b,t,4,d//4'))
```

Similarly, use `tsalib.permute_transform` to compute permutation index order from a declarative spec. 
```python 
    perm_indices = permute_transform(src=(B,T,D), to=(D,T,B)) #(2, 1, 0)
    x = x.transpose(perm_indices) #(10, 50, 300) -> (300, 50, 10)
    
    #or, compact:
    x = x.transpose(pt('btd','dtb'))
```

Use dimension names instead of cryptic indices.
```python
ax = (2, B, D).index(B) #ax = 1
c: (2, D) = np.mean(b, axis=ax) 
print(f'after mean along axis {B}={ax}: {(2,D)}: {c.shape}') #... axis Batch=1: (2, EmbedDim): (2, 3)
```

See [tests/test_ext.py](tests/test_ext.py) for complete examples.


## Examples

 The [examples](examples) directory contains TS annotations of a few well-known, complex neural architectures: [resnet](examples/resnet.py), [transformer](examples/openai_transformer.py). With TSAs, we can gain deeper and immediate insight into how the module works by scanning through the `forward` function.

## Dependencies

`sympy`. A library for building symbolic expressions in Python.

Tested with Python 2.7, 3.6. For writing type annotations inline, Python >= 3.5 is required.

Python >= 3.5 allows optional type annotations for variables. These annotations do not affect the program performance in any way. 





## References

* Blog [article](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b) introducing TSA.
* A [proposal](https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY/edit#heading=h.rkj7d39awayl) for designing a tensor library with named dimensions from ground-up. The TSA library takes care of some use cases, without requiring any change in the tensor libraries.
* Pytorch Issue on Names Axes [here](https://github.com/pytorch/pytorch/issues/4164).
* Using [einsum](http://ajcr.net/Basic-guide-to-einsum/) for tensor operations improves productivity and code readability. [blog](https://rockt.github.io/2018/04/30/einsum)
* The [Tile](https://vertexai-plaidml.readthedocs-hosted.com/en/latest/writing_tile_code.html) DSL uses indices ranging over dimension variables to write compact, library-independent tensor operations.
* The [datashape](https://datashape.readthedocs.io/en/latest/) library introduces a generic type system and grammar for structure data. `tsalib` focuses on shapes of homogeneous tensor data types only, with arithmetic support.

## Contributors

Nishant Sinha, OffNote Labs. @[medium](https://medium.com/@ekshakhs), @[twitter](https://twitter.com/ekshakhs)


