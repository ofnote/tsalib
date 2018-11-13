# Tensor Shape Annotations Library (tsalib)

Writing deep learning programs which manipulate multi-dimensional tensors (`numpy`, `pytorch`, `keras`, `tensorflow`, ...) requires you to carefully keep track of shapes of matrices/tensors. The Tensor Shape Annotation (TSA) library enables you to write first-class, library-independent, **shape expressions** over **dimension variables** to model matrix/tensor variable shapes.
TSAs enable us to label and verify tensor variables shapes as well as write more *fluent* shape transformations and tensor operations. Using TSAs enhances code clarity, accelerates debugging and improves overall developer productivity when writing tensor programs. 
Detailed article [here](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b).

See updates [here](#change-log).

## Introduction

 Carrying around the tensor shapes in your head gets increasingly hard as programs become more complex, e.g., reshaping before a `matmult`, examining/modifying deep pre-trained architectures (`resnet`, `densenet`, `elmo`), designing new kinds of `attention` mechanisms (`multi-head attention`) or when creating a new `RNN` cell. There is no principled way of shape specification and tracking inside code -- most developers resort to writing adhoc comments embedded in code to keep track of tensor shapes (see code from [google-research/bert](https://github.com/google-research/bert/blob/a21d4848ec33eca7d53dd68710f04c4a4cc4be50/modeling.py#L664)).

`tsalib` comes to our rescue here. It allows you to write shape expressions over dimension variables describing the shape of tensor variables. These expressions can be used in multiple ways: 
- as first-class annotations of tensor variables,
- to write `symbolic` shape `assert`ions and tensor constructors
- to specify shape transformations (`reshape`, `permute`, `expand`) or tensor product operations (`matmult`) succinctly. 

TSAs expose the typically *invisible* tensor shape types, leading to improved productivity across the board. 

## Dimension Variables

Tensor shape annotations (TSAs) are constructed using `dimension` variables --`B` (Batch), `C` (Channels), `D` (EmbedDim) -- and arithmetic expressions (`B*2`, `C+D`) over them. Using `tsalib`, you can define dimension variables customized to your architecture/program.

TSAs may be be represented as
* a tuple `(B,H,D)` [long form]
* a string `'b,h,d'` (compact notation) (or simply `'bhd'`)

Here is an example snippet which uses TSAs in a `pytorch` program to define, transform and verify tensor shapes. TSAs work seamlessly with arbitrary tensor libraries:  `numpy`, `pytorch`, `keras`, `tensorflow`, `mxnet`, etc.

```python
from tsalib import dim_vars as dvs
from tsalib import view_transform as vt

#declare dimension variables
B, C, H, W = dvs('Batch:32 Channels:3 Height:256 Width:256') 
...
#create tensors using dimension variables (interpret dim vars as integers)
x: (B, C, H, W) = torch.randn(B, C, H, W) 
#perform tensor transformations
x: (B, C, H // 2, W // 2) = maxpool(x) 
#check symbolic assertions over TSAs, without knowing concrete shapes
assert x.size() == (B, C, H // 2, W // 2)

#reshape/permute using shorthand (einsum-like) notation
x1 = x.view(vt(',,kl', ',,k*l', x.size()))
assert x1.size() == (B, C, (H//2)*(W//2))
# altneratively : super convenient reshapes!
x2 = x.view ((B,C, (H//2)*(W//2)))


``` 

Shape annotations/assertions turn out to be useful in many ways. 
* They help us to quickly verify the variable shapes when writing new transformations or modifying existing modules. 
* Assertions and annotations remain the same even if the concrete dimension lengths change.
* Faster *debugging*: if you annotate-as-you-go, the tensor variable shapes are explicit in code, readily available for a quick inspection. No more adhoc shape `print`ing when investigating obscure shape errors.
* Do shape transformations using *shorthand* notation and avoid unwanted shape surgeries.
* Use TSAs to improve code clarity everywhere, even in your machine learning data pipelines.
* They serve as useful documentation to help others understand or extend your module.


## Installation

`pip install tsalib`

## Getting Started

See [tests/test.py](tests/test.py) and [tests/test_ext.py](tests/test_ext.py) for complete examples of basic and extended usage.

```python
from tsalib import dim_var as dv, dim_vars as dvs, dim_vars_shape as dvs2
import numpy as np
```

### Declare Dimension Variables, Expressions over them
```python
B, C, D, H, W = dv('Batch'), dv('Channels'), dv('EmbedDim'), dv('Height'), dv('Width')
#or
B, C, D, H, W = dvs('Batch Channels EmbedDim Height Width')
#or declare dim vars with default integer values (optional)
B, C, D, H, W = dvs('Batch:48 Channels:3 EmbedDim:300 Height Width')
#or provide *shorthand* names for dim vars
B, C, D, H, W = dvs('Batch(b):48 Channels(c):3 EmbedDim(d):300 Height(h) Width(w)')

# switch from using config constants to using dimension vars
B, C, D = dvs2('Batch(b) Channels(c) EmbedDim(d)', (config.batch_size, config.num_channels, config.embed_dim))

# TSAs are tuples over dimension variables
S1 = (B, C, D)
# we can always verify TSAs against concrete shapes
assert S1 == (48, 3, 300)
```

### Use Dimension Variables to declare Tensors

Instead of scalar variables `batch_size`, `embed_dim`, use dimension variables `B`, `D` uniformly throughout your code.

```python
B, D = dvs('Batch:48 EmbedDim:300')
#declare a 2-D tensor of shape(48, 300)
x = torch.randn(B, D)
#assertions over dimension variables (not exact values)
assert x.size() == (B, D)
```


### Use TSAs to annotate variables on-the-go (Python 3)

```python
a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]]) #(Batch, EmbedDim): (2, 3)

b: (2, B, D) = np.stack([a, a]) #(2, Batch, EmbedDim): (2, 2, 3)
```

Arithmetic over dimension variables is supported. This enables easy tracking of shape changes across neural network layers.

```python
v: (B, C, H, W) = torch.randn(B, C, h, w)
x : (B, C * 2, H//2, W//2) = torch.nn.conv2D(C, C*2, ...)(v) 
```

### Use TSAs to make matrix operations compact and explicit


Avoid explicit shape computations for `reshaping`. 
```python
    #use dimension variables directly
    x = torch.ones(B, T, D)
    x = x.view(B, T, 4, D//4)
```

In general, use `tsalib.view_transform` to specify view changes declaratively.

```python
    x = np.ones((B, T, D))
    from tsalib import view_transform as vt
    #or, compact form:
    x = x.reshape(vt('btd', 'b,t,4,d//4', x.shape)) #(20, 10, 300) -> (20, 10, 4, 75)
    #or, super-compact, using anonymous dimensions:
    x = x.reshape(vt(',,d', ',,4,d//4', x.shape))
```

Similarly, use `tsalib.permute_transform` to compute permutation index order (no manual guess-n-check) from a declarative spec.
```python 
    # long form:
    perm_indices = permute_transform(src=(B,T,D,K), to=(D,T,B,K)) #(2, 1, 0, 3)
    x = x.transpose(perm_indices) #(10, 50, 300, 30) -> (300, 50, 10, 30)
    
    from tsalib import permute_transform as pt
    #or, compactly:
    x = x.transpose(pt('btdk', 'dtbk'))
    #or, super-compact:
    x = x.transpose(pt('b,,d,', 'd,,b,'))

```

Use dimension names instead of cryptic indices.
```python
ax = (2, B, D).index(B) #ax = 1
c: (2, D) = np.mean(b, axis=ax) 
print(f'after mean along axis {B}={ax}: {(2,D)}: {c.shape}') #... axis Batch=1: (2, EmbedDim): (2, 3)
```


See [tests/test.py](tests/test.py) and [tests/test_ext.py](tests/test_ext.py) for complete examples of basic and extended usage.


## Examples

 The [examples](examples) directory contains TS annotations of a few well-known, complex neural architectures: [Resnet](examples/resnet.py), [OpenAI Transformer](examples/openai_transformer.py). With TSAs, we can gain deeper and immediate insight into how the module works by scanning through the `forward` function.

## Dependencies

`sympy`. A library for building symbolic expressions in Python.

Tested with Python 3.6. For writing type annotations inline, Python >= 3.5 is required.

Python >= 3.5 allows optional type annotations for variables. These annotations do not affect the program performance in any way. 


## Best Practices

* Convert all *relevant* config parameters into dimension variables. Use only the latter in your code.
* Avoid using `reshape` : use `view` and `transpose` together. An inadvertent `reshape` may not preserve your dimensions (axes). Using `view` to change shape protects against this: it throws an error if the dimensions being manipulated are not contiguous. 


## References

* Blog [article](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b) introducing TSA.
* A [proposal](https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY/edit#heading=h.rkj7d39awayl) for designing a tensor library with named dimensions from ground-up. The TSA library takes care of some use cases, without requiring any change in the tensor libraries.
* Pytorch Issue on Names Axes [here](https://github.com/pytorch/pytorch/issues/4164).
* Using [einsum](http://ajcr.net/Basic-guide-to-einsum/) for tensor operations improves productivity and code readability. [blog](https://rockt.github.io/2018/04/30/einsum)
* The [Tile](https://vertexai-plaidml.readthedocs-hosted.com/en/latest/writing_tile_code.html) DSL uses indices ranging over dimension variables to write compact, library-independent tensor operations.
* The [datashape](https://datashape.readthedocs.io/en/latest/) library introduces a generic type system and grammar for structure data. `tsalib` focuses on shapes of homogeneous tensor data types only, with arithmetic support.

## Contributors

Nishant Sinha, OffNote Labs. @[medium](https://medium.com/@ekshakhs), @[twitter](https://twitter.com/ekshakhs)

## Change Log

* [9 Nov 2018] Support for shorthand notation in view/permute/expand transforms.
* [9 Nov 2018] Support for using TSA in assertions and tensor constructors (cast to integers).
* [25 Oct 2018] Initial Release


