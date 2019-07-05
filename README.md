# Tensor Shape Annotations Library (tsalib)
[![PyPI version](https://badge.fury.io/py/tsalib.svg)](https://badge.fury.io/py/tsalib)
[![Chat](https://img.shields.io/gitter/room/offfnote/tsalib.svg?colorB=yellow&style=plastic)](https://gitter.im/offfnote/tsalib)

Conventional tensor manipulation libraries — `numpy`, `pytorch`, `keras`, `tensorflow`, lack support for *naming* the dimensions of tensor variables. `tsalib` enables using *named dimensions* with existing libraries, using Python's support for *type annotations* and a new *shorthand notation* for naming tensor shapes (**TSN**). 

Why *named dimensions* ?  See [References](#References).

---
Using `tsalib`:

* track shapes: label your tensor variables with their **named shapes** (`x: 'btd'` or `x: (B,T,D)`)

* better debugging: write *named* shape **assertions** (`assert x.shape == (B,T,D)`).

* write seamless *named* shape **transformations**:

  `warp(x, '(btd)* -> btdl -> bdtl -> b,d//2,t*2,l', 'jpv')`

  instead of a sequence of calls over a laundry list of APIs (`reshape`,`permute`,`stack`, `concat`)

* work with arbitrary backends without changes:  `numpy`, `pytorch`, `keras`, `tensorflow`, `mxnet`, etc.

Exposing the *invisible* named dimensions enhances code clarity, accelerates debugging and leads to improved productivity across the board. Even complex deep learning architectures need only a small number of named dimensions.

The complete **API** in a **notebook** [here](notebooks/tsalib.ipynb), an introductory **article** [here](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b).

---
## Contents

- [Quick Start](#Quick-Start)
- [Installation](#Installation) 
- [Design Principles, Model Examples](#Documentation-Design-Principles-Model-Examples) (includes [BERT](models/bert)!)
- [API Overview](#API)
- [Best Practices -- How to use `tsalib`](#Best-Practices)
- [Change Log](#change-log)



### Quick Start

```python
from tsalib import dim_vars as dvs, size_assert
import tensorflow as tf
import torch

#declare dimension variables (from config arguments)
B, C, H, W = dvs('Batch(b):32 Channels(c):3 Height(h):256 Width(w):256') 
...
# create tensors (pytorch) using dimension variables (interpret dim vars as integers)
x: 'bchw' = torch.randn(B, C, H, W)
x: 'bchw' = tf.get_variable("x", shape=(B, C, H, W), initializer=tf.random_normal_initializer())

# perform tensor transformations, keep track of named shapes
x: 'b,c,h//2,w//2' = maxpool(x) 

# check assertions: compare dynamic shapes with declared shapes
# assertions are 'symbolic': don't change even if declared shapes change
assert x.size() == (B, C, H // 2, W // 2)
#or, check selected dimensions
size_assert (x.size(), (B,C,H//2,W//2), dims=[1,2,3])
```

Write intuitive and crisp shape transformations:

```python
from tsalib import permute_transform as pt

# permute: irrelevant dimensions are anonymous (underscores).
x: 'bchw'
x1 = x.permute(pt('_c__ -> ___c'))
assert x1.size() == (B, H, W, C)

# A powerful one-stop `warp` operator to compose multiple transforms inline
# here: a sequence of a permute ('p') and view ('v') transformations
y = warp(x1, 'bhwc -> bchw -> b*c,h,w', 'pv')
assert y.size() == (B*C,H,W)

#or, the same transformation sequence with anonymous dims
y = warp (x1, ['_hwc -> _chw', 'bc,, -> b*c,,'], 'pv')

# Combinations of `alignto`, `dot` and broadcast
# Enables writing really compact code for similar patterns
ht: 'bd'; Wh: 'dd'; Y: 'bld'; WY: 'dd'

a: 'bd' = dot('_d.d_', ht, Wh) 
b: 'b,1,d' = alignto((a,'bd'), 'bld')
Mt: 'bld' = torch.tanh(dot('__d.d_', Y, WY) + b)

```



**Old vs New code**

```python
def merge_heads_old(x):
  x = x.permute(0, 2, 1, 3).contiguous()
  new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
  res = x.view(*new_x_shape)
```

```python
def merge_heads_tsalib(x: 'bhtd'):
    res: 'b,t,h*d' = warp(x, 'bhtd -> bthd -> b,t,h*d', 'pcv')
```



Named shapes may be represented as tuples or *shorthand* strings:

- a tuple `(B,H,D)` [long form] or a string `'b,h,d'` (or simply `'bhd'`) [shorthand]
- a string with anonymous dimensions (`',h,'` or `_h_` is a 3-d tensor).

More details on shorthand notation [here](notebooks/shorthand.md) . 

## Installation

`pip install [--upgrade] tsalib`

## Documentation, Design Principles, Model Examples

This [notebook](notebooks/tsalib.ipynb) serves as a working documentation for the `tsalib` library and illustrates the complete `tsalib` API. The **shorthand** notation is documented [here](notebooks/shorthand.md).

The [models](models) directory contains tsalib annotations of a few well-known, complex neural architectures: 
- [BERT](models/bert). 
- [OpenAI Transformer](models/openai_transformer.py),
- [Resnet](models/resnet.py),
- Contrast models with and without tsalib ([pytorch](models/snippets_pytorch.py), [tensorflow](models/snippets_tf.py)).

With TSAs, we can gain deeper and immediate insight into how the module works by scanning through the `forward` (or equivalent) function.
- `tsalib` is designed to stay light and easy to incorporate into existing workflow with minimal code changes. Choose to use `tsalib` for tensor labels and shape asserts only, or, integrate deeply by using `warp` everywhere in your code.
- The API includes both library-independent and dependent parts, giving developers flexibility in how they choose to incorporate `tsalib` in their workflow.
- We've carefully avoided deeper integration into popular tensor libraries to keep `tsalib` light-weight and avoid backend-inflicted bugs.



## API

```python
from tsalib import dim_vars as dvs, get_dim_vars
import numpy as np
```
### Declarations

#### Declare Dimension Variables
```python
#or declare dim vars with default integer values (optional)
B, C, D, H, W = dvs('Batch:48 Channels:3 EmbedDim:300 Height Width')
#or provide *shorthand* names and default values for dim vars [best practice]
B, C, D, H, W = dvs('Batch(b):48 Channels(c):3 EmbedDim(d):300 Height(h) Width(w)')

# switch from using config arguments to named dimensions
B, C, D = dvs('Batch(b):{0} Channels(c):{1} EmbedDim(d):{2}'.format(config.batch_size, config.num_channels, config.embed_dim))

```

#### Declare Tensors and Annotate Shapes

Instead of scalar variables `batch_size`, `embed_dim`, use dimension variables `B`, `D` or shorthands `b,d` throughout your code.

```python
B, D = dvs('Batch(b):{batch_size} EmbedDim(d):{embed_dim}}')
#declare a 2-D tensor of shape(48, 300)
x = torch.randn(B, D)
#assertions over dimension variables (code unchanged even if dim sizes change)
assert x.size() == (B, D)
```



```python
B, D = get_dim_vars('b d') #lookup pre-declared dim vars
a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]]) #(Batch, EmbedDim): (2, 3)
b: (2, B, D) = np.stack([a, a]) #(2, Batch, EmbedDim): (2, 2, 3)

#or simply, use shorthand shapes
a: 'b,d'
b: '2bd'
```
Annotations are optional and do not affect program performance. Arithmetic over dimension variables is supported. This enables easy tracking of shape changes across neural network layers.

```python
B, C, H, W = get_dim_vars('b c h w') #lookup pre-declared dim vars
v: 'bchw' = torch.randn(B, C, h, w)
x : 'b,c*2,h//2,w//2' = torch.nn.conv2D(C, C*2, ...)(v) 
```
### Shape and Tensor Transformations

#### One-stop shape transforms: `warp` operator

The `warp` operator enables squeezing in a **sequence** of shape transformations in a single line using [TSN](notebooks/shorthand.md). The operator takes in an input tensor, a sequence of shape transformations, and the corresponding transform types (view transform -> 'v', permute transform -> 'p'). See docs for transform types [here](notebooks/shorthand.md#warp-transformation).

```python
    x: 'btd' = torch.randn(B, T, D)
    y = warp(x, 'btd -> b,t,4,d//4 ->  b,4,t,d//4 ', 'vp') #(v)iew, then (p)ermute, transform
    assert(y.shape == (B,4,T,D//4))
```
Because it returns transformed tensors, the `warp` operator is backend library-dependent. Currently supported backends are `numpy`, `tensorflow` and `pytorch`. New backends can be added easily (see [backend.py](tsalib/backend.py)).

Or, use individual named shape transformations:

```
	#use dimension variables directly
	x = torch.ones(B, T, D)
	x = x.view(B, T, 4, D//4)
	from tsalib import view_transform as vt, permute_transform as pt
	y = x.reshape(vt('btd -> b,t,4,d//4', x.shape)) #(20, 10, 300) -> (20, 10, 4, 75)
	assert y.shape == (B, T, 4, D//4)
	y = x.transpose(pt('b,,d, -> d,,b,'))
```

See [notebook](notebooks/tsalib.ipynb) for more examples.

#### More useful operators: `join`, `alignto`, `reduce_dims` ...
<details>
    <summary>more ..</summary>

Unified `stack/concat` using `join`. Join together sequence of tensors into a single tensor in different ways using the same `join` operator. `join` is backend-dependent.

```python
    # xi : (B, T, D)
    # "concatenate" along the 'T' dimension: "(b,t,d)* -> (b,3*t,d)"
    x = tsalib.join([x1, x2, x3], ',*,') 
    assert x.shape == (B, 3*T, D)

    # "stack": join by adding a new dimension to the front: "(b,t,d)* -> (^,b,t,d)"
    x = join([x1, x2, x3], '^') 
    assert x.shape == (3, B, T, D)

```

Align one tensor to the rank of another tensor using `alignto`.

```python
    x1 = np.random.randn(D,D)
    x2 = np.random.randn(B,D,T,D)

    x1_aligned = alignto((x1, 'dd'), 'bdtd')
    assert x1_aligned.shape == (1,D,1,D)
    x1_aligned = alignto((x1, 'dd'), 'bdtd', tile=True)
    assert x1_aligned.shape == (B,D,T,D)
```


Use dimension names instead of cryptic indices in *reduction* (`mean`, `max`, ...) operations.
```python
    from tsalib import reduce_dims as rd
    b: (2, B, D)
    c: (D,) = np.mean(b, axis=rd('2bd -> d')) #axis = (0,1)
```

</details>

#### Simplified `dot` operator

Easy `matmult` specification when 
- exactly a single dimension is common between the operands and 
- the order of dimensions preserved in the output.

```python
    x = torch.randn(B, C, T)
    y = torch.randn(C, D)
    z = dot('_c_.c_', x, y)
    assert z.size() == (B, T, D)
```


## Dependencies

`sympy`. A library for building symbolic expressions in Python is the only dependency.

Tested with Python 3.6. Core API should work with Python 2. Contributions welcome.

For writing type annotations inline, Python >= 3.5 is required which allows optional type annotations for variables. These annotations do not affect the program performance in any way. 


## Best Practices

* `tsalib` is designed for **progressive adoption** with your current deep learning models and pipelines. You can start off only with declaring and labeling variables with named shapes and writing shape assertions. This already brings tremendous improvement in productivity and code readability. Once comfortable, use other transformations: `warp`, `join`, etc.
* Convert all *relevant* config parameters into dimension variables. Use only latter in your code.
* Define all dimension variables upfront -- this requires some discipline. Use `get_dim_vars` to lookup pre-defined dimension variables by their shorthand names in any function context.
* Avoid using `reshape` : use `view` and `transpose` together. An inadvertent `reshape` may not preserve your dimensions (axes). Using `view` to change shape protects against this: it throws an error if the dimensions being manipulated are not contiguous. 
* Shape *Annotations* vs *Assertions*. Shape labels (`x: (B,T,D)` or `x: 'btd'`) ease shape recall during coding. Shape assertions (`assert x.shape === (B,T,D)`) enable catching inadvertent shape bugs at runtime. Pick either or both to work with.


## References

* Blog [article](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b) introducing TSA.
* A [proposal](https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY/edit#heading=h.rkj7d39awayl) for designing a tensor library with named dimensions from ground-up. The TSA library takes care of some use cases, without requiring any change in the tensor libraries.
* Pytorch Issue on Names Axes [here](https://github.com/pytorch/pytorch/issues/4164).
* Using [einsum](http://ajcr.net/Basic-guide-to-einsum/) for tensor operations improves productivity and code readability. [blog](https://rockt.github.io/2018/04/30/einsum)
* The [Tile](https://vertexai-plaidml.readthedocs-hosted.com/en/latest/writing_tile_code.html) DSL uses indices ranging over dimension variables to write compact, library-independent tensor operations.
* The [datashape](https://datashape.readthedocs.io/en/latest/) library introduces a generic type system and grammar for structure data. `tsalib` focuses on shapes of homogeneous tensor data types only, with arithmetic support.
* The [xarray](https://github.com/pydata/xarray) library.
* The [einops](https://github.com/arogozhnikov/einops) library.
* The [NamedTensor](http://nlp.seas.harvard.edu/NamedTensor) library.
* The [TensorNetwork](https://github.com/google/TensorNetwork) library. Generalizes the idea of named axes and composition/decomposition/reordering of axes very nicely.



Writing deep learning programs which manipulate multi-dim tensors (`numpy`, `pytorch`, `tensorflow`, ...) requires you to carefully keep track of shapes of tensors. In absence of a principled way to *name* tensor dimensions and track shapes, most developers resort to writing adhoc shape comments embedded in code (see code from [google-research/bert](https://github.com/google-research/bert/blob/a21d4848ec33eca7d53dd68710f04c4a4cc4be50/modeling.py#L664)) or spaghetti code with numeric indices: `x.view(* (x.size()[:-2] + (x.size(-2) * x.size(-1),))`. This makes both reading — figuring out `RNN` output shapes, examining/modifying deep pre-trained architectures (`resnet`, `densenet`, `elmo`) — and writing —  designing new kinds of `attention` mechanisms (`multi-head attention`)— deep learning programs harder.

<details>
    <summary>
    Developers benefit from shape annotations/assertions in many ways: 
    </summary>
    Benefits:
    * Quickly verify the variable shapes when writing new transformations or modifying existing modules. 
    * Assertions and annotations remain the same even if the actual dimension sizes change.
    * Faster *debugging*: if you annotate-as-you-go, the tensor variable shapes are explicit in code, readily available for a quick inspection. No more adhoc shape `print`ing when investigating obscure shape errors.
    * Do shape transformations using *shorthand* notation and avoid unwanted shape surgeries.
    * Use TSAs to improve code clarity everywhere, even in your machine learning data pipelines.
    * They serve as useful documentation to help others understand or extend your module.
</details>

## 

## Author

 * Nishant Sinha, [OffNote Labs](http://offnote.co) (nishant@offnote.co, @[medium](https://medium.com/@ekshakhs), @[twitter](https://twitter.com/ekshakhs))

## Change Log
The library is in its early phases. Contributions/feedback welcome!

* [5 Feb 2019] Added `dot` operator.
* [4 Feb 2019] Added fully annotated and adapted BERT [model](models/bert). More illustrative pytorch and tensorflow snippets.
* [31 Jan 2019] Added `alignto` operator.
* [18 Dec 2018] Added the `join` operator. `warp` takes a list of (shorthand) transformations.
* [28- Nov 2018] Added `get_dim_vars` to lookup dim vars declared earlier. Shorthand notation docs.
* [21 Nov 2018] Added documentation [notebook](notebooks/tsalib.ipynb). 
* [18 Nov 2018] Support for `warp`, `reduce_dims`. Backend modules for `numpy`, `tensorflow` and `torch` added.
* [9 Nov 2018] Support for shorthand notation in view/permute/expand transforms.
* [9 Nov 2018] Support for using TSA in assertions and tensor constructors (cast to integers).
* [25 Oct 2018] Initial Release


