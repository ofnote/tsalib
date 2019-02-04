# Tensor Shape Annotations Library (tsalib)

Writing deep learning programs which manipulate multi-dim tensors (`numpy`, `pytorch`, `keras`, `tensorflow`, ...) requires you to carefully keep track of shapes of tensors. In absence of a principled way to *name* tensor dimensions and track shapes, most developers resort to writing adhoc shape comments embedded in code (see code from [google-research/bert](https://github.com/google-research/bert/blob/a21d4848ec33eca7d53dd68710f04c4a4cc4be50/modeling.py#L664)).

The `tsalib` library enables you to write 
- first-class, library-independent, shape annotations (TSAs) over **named dimension variables**,
- defensive **shape assertions** using these named shapes, and,
- more *fluent* shape **transformations** and tensor **operations** using tensor shorthand notation (**TSN**).

TSAs expose the typically *invisible* tensor dimension names, which enhances code clarity, accelerates debugging and leads to improved productivity across the board. 

The `tsalib` API **notebook** is [here](notebooks/tsalib.ipynb). Detailed article [here](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b).

## Contents

- [Quick Start -- Dimension Variables, Tensor Shorthand Notation](#Dimension-Variables)
- [Installation](#Installation) 
- [Design Principles, Model Examples](#Documentation-Design-Principles-Model-Examples)
- [API Overview](#API)
- [Best Practices -- How to use `tsalib`](#Best-Practices)
- [Change Log](#change-log)


<details>
    <summary> <b>Why tsalib?</b> Carrying around the tensor shapes in your head gets increasingly hard as programs become more complex. ...
    </summary>
    For example, reshaping before a `matmult`, figuring out `RNN` output shapes, examining/modifying deep pre-trained architectures (`resnet`, `densenet`, `elmo`), designing new kinds of `attention` mechanisms (`multi-head attention`). 
    `tsalib` comes to our rescue here. It allows you to write symbolic shape expressions over dimension variables describing tensor variable shapes. These expressions can be used in multiple ways: 
    - as first-class annotations of tensor variables,
    - to write `symbolic` shape `assert`ions and tensor constructors
    - to specify shape transformations (`reshape`, `permute`, `expand`) succinctly. 
</details>


<details>
    <summary>
    Developers benefit from shape annotations/assertions in many ways: (more ..)
    </summary>
    * Quickly verify the variable shapes when writing new transformations or modifying existing modules. 
    * Assertions and annotations remain the same even if the actual dimension sizes change.
    * Faster *debugging*: if you annotate-as-you-go, the tensor variable shapes are explicit in code, readily available for a quick inspection. No more adhoc shape `print`ing when investigating obscure shape errors.
    * Do shape transformations using *shorthand* notation and avoid unwanted shape surgeries.
    * Use TSAs to improve code clarity everywhere, even in your machine learning data pipelines.
    * They serve as useful documentation to help others understand or extend your module.
</details>


## Dimension Variables

Tensor shape annotations (TSAs) are constructed using `dimension` variables --`B` (Batch), `C` (Channels), `D` (EmbedDim) -- and arithmetic expressions (`B*2`, `C+D`) over them. Using `tsalib`, you can define dimension variables customized to your architecture/program. Even complex architectures need only a small number of named dimensions.

TSAs may be represented as tuples or *shorthand* strings:
* a tuple `(B,H,D)` [long form]
* a string `'b,h,d'` (or simply `'bhd'`)
* a string with anonymous dimensions (`',h,'` is a 3-d tensor).

The tensor shorthand notation ([TSN](notebooks/shorthand.md)), is used extensively in tsalib.

Here is an example snippet which uses TSAs and TSNs in a `pytorch` program to define, transform and verify tensor shapes. `tsalib` is designed to work seamlessly with arbitrary backends:  `numpy`, `pytorch`, `keras`, `tensorflow`, `mxnet`, etc.

```python
from tsalib import dim_vars as dvs
from tsalib import view_transform as vt, permute_transform as pt

#declare dimension variables
B, C, H, W = dvs('Batch:32 Channels:3 Height:256 Width:256') 
...
# create tensors (pytorch) using dimension variables (interpret dim vars as integers)
x: (B, C, H, W)=torch.randn( (B, C, H, W) )
# or use shorthand labels
x: 'bchw'=tf.get_variable("v", shape=(B, C, H, W), initializer=tf.random_normal_initializer())

# perform tensor transformations
x: (B, C, H // 2, W // 2) = maxpool(x) 

# check symbolic assertions over TSAs
# assertions don't change even if dim sizes change
assert x.size() == (B, C, H // 2, W // 2)
#or, check selected dimensions
size_assert (x.size(), (B,C,H//2,W//2), dims=[1,2])

# super convenient reshapes (long form)!
x1 = x.view ((B, C, (H//2)*(W//2)))
assert x1.size() == (B, C, (H//2)*(W//2))

```

Use tensor shorthand notation ([TSN](notebooks/shorthand.md)) to write intuitive and quick shape changes.

```python

# permute: irrelevant dimensions are anonymous (underscores).
x: (B, C, H, W)
x1 = x.permute(pt('_c__ -> ___c'))
assert x1.size() == (B, H, W, C)

# Writing multiple shape transforms one-by-one gets cumbersome
# A powerful one-stop `warp` operator to compose multiple transforms inline

# here: a sequence of a permute ('p') and view ('v') transformations
y = warp(x1, 'bhwc -> bchw -> b*c,h,w', 'pv')
assert y.size() == (B*C,H,W)

#or, with anonymous dims
y = warp (x1, ['_hwc -> _chw', 'bc,, -> b*c,,'], 'pv')

``` 

## Installation

`pip install [--upgrade] tsalib`

## Documentation, Design Principles, Model Examples

This [notebook](notebooks/tsalib.ipynb) serves as a working documentation for the `tsalib` library and illustrates the complete `tsalib` API. The **shorthand** notation is documented [here](notebooks/shorthand.md).

- `tsalib` is designed to stay light and easy to incorporate into existing workflow with minimal code changes. Choose to use `tsalib` for tensor labels and shape asserts only, or, integrate deeply by using `warp` everywhere in your code.
- The API includes both library-independent and dependent parts, giving developers flexibility in how they choose to incorporate `tsalib` in their workflow.
- We've carefully avoided deeper integration into popular tensor libraries to keep `tsalib` light-weight and avoid backend-inflicted bugs.


The [models](models) directory contains tsalib annotations of a few well-known, complex neural architectures: [Resnet](models/resnet.py), [OpenAI Transformer](models/openai_transformer.py). With TSAs, we can gain deeper and immediate insight into how the module works by scanning through the `forward` function.



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
#or provide optional *shorthand* names for dim vars, default values
B, C, D, H, W = dvs('Batch(b):48 Channels(c):3 EmbedDim(d):300 Height(h) Width(w)')

# switch from using config constants to using dimension vars
B, C, D = dvs('Batch(b):{0} Channels(c):{1} EmbedDim(d):{2}'.format(config.batch_size, config.num_channels, config.embed_dim))

```

#### Use Dimension Variables to declare Tensors

Instead of scalar variables `batch_size`, `embed_dim`, use dimension variables `B`, `D` uniformly throughout your code.

```python
B, D = dvs('Batch:{batch_size} EmbedDim:{embed_dim}}')
#declare a 2-D tensor of shape(48, 300)
x = torch.randn(B, D)
#assertions over dimension variables (code unchanged even if dim sizes change)
assert x.size() == (B, D)
```


#### Use TSAs to annotate variables on-the-go (Python 3)

```python
B, D = get_dim_vars('b d') #lookup pre-declared dim vars
a: (B, D) = np.array([[1., 2., 3.], [10., 9., 8.]]) #(Batch, EmbedDim): (2, 3)

b: (2, B, D) = np.stack([a, a]) #(2, Batch, EmbedDim): (2, 2, 3)
```
Annotations are optional and do not affect program performance.

Arithmetic over dimension variables is supported. This enables easy tracking of shape changes across neural network layers.

```python
B, C, H, W = get_dim_vars('b c h w') #lookup pre-declared dim vars
v: (B, C, H, W) = torch.randn(B, C, h, w)
x : (B, C * 2, H//2, W//2) = torch.nn.conv2D(C, C*2, ...)(v) 
```
### Shape and Tensor Transformations

#### Reshape, Permute/Transpose transformations 

Avoid explicit shape computations for `reshaping`. 
```python
    #use dimension variables directly
    x = torch.ones(B, T, D)
    x = x.view(B, T, 4, D//4)
```

<details>
<summary>In general, use `tsalib.view_transform` to specify view changes declaratively. ... </summary>

```python
    x = np.ones((B, T, D))
    from tsalib import view_transform as vt
    #or, compact form:
    y = x.reshape(vt('btd -> b,t,4,d//4', x.shape)) #(20, 10, 300) -> (20, 10, 4, 75)
    assert y.shape == (B, T, 4, D//4)
    #or, super-compact, using anonymous dimensions:
    y = x.reshape(vt(',,d -> ,,4,d//4', x.shape))
```
</details>

<details>
<summary>Similarly, use `tsalib.permute_transform` to compute permutation index order (no manual guess-n-check) from a declarative spec. ... </summary>

```python 
    from tsalib import permute_transform as pt

    x = np.ones ((B, T, D, K))
    perm_indices = pt('btdk -> dtbk') # (2, 1, 0, 3)
    y = x.transpose(perm_indices)
    assert y.shape == (D, T, B, K)

    #or, super-compact:
    y = x.transpose(pt('b,,d, -> d,,b,'))

```
</details>



#### One-stop shape transforms: `warp` operator

The `warp` operator allows squeezing in a **sequence** of shape transformations in a single line using [TSN](notebooks/shorthand.md). The operator takes in an input tensor, a sequence of shape transformations, and the corresponding transform types (view transform -> 'v', permute transform -> 'p'). See docs for transform types [here](notebooks/shorthand.md#warp-transformation).

```python
    x: 'btd' = torch.randn(B, T, D)
    y = warp(x, 'btd -> b,t,4,d//4 ->  b,4,t,d//4 ', 'vp') #(v)iew, then (p)ermute, transform
    assert(y.shape == (B,4,T,D//4))
```
Because it returns transformed tensors, the `warp` operator is backend library-dependent. Currently supported backends are `numpy`, `tensorflow` and `pytorch`. New backends can be added easily (see [backend.py](tsalib/backend.py)).

See [notebook](notebooks/tsalib.ipynb) for complete working examples.

#### `join`, `alignto`, `reduce_dims` ...
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

    x1_aligned = alignto((x1, 'dd'), (x2, 'bdtd'))
    assert x1_aligned.shape == (B,D,T,D)
    x1_aligned = alignto((x1, 'dd'), (x2, 'bdtd'), expand=False)
    assert x1_aligned.shape == (1,D,1,D)
```


Use dimension names instead of cryptic indices in *reduction* (`mean`, `max`, ...) operations.
```python
    from tsalib import reduce_dims as rd
    b: (2, B, D)
    c: (D,) = np.mean(b, axis=rd('2bd -> d')) #axis = (0,1)
```

</details>



## Dependencies

`sympy`. A library for building symbolic expressions in Python is the only dependency.

Tested with Python 3.6. Core API should work with Python 2. Contributions welcome.

For writing type annotations inline, Python >= 3.5 is required which allows optional type annotations for variables. These annotations do not affect the program performance in any way. 


## Best Practices

* `tsalib` is designed for **progressive adoption** with your current deep learning models and pipelines. You can start off only with declaring dimension variables, labeling statements with TSAs and writing shape assertions. This already brings tremendous improvement in productivity and code readability. Once comfortable, move on to using the advanced features of tsalib: shorthand shape transformations, warp, join, etc.
* Convert all *relevant* config parameters into dimension variables. Use only latter in your code.
* Define all dimension variables upfront -- this requires some discipline. Use `get_dim_vars` to lookup pre-defined dimension variables by their shorthand names in any function context.
* Avoid using `reshape` : use `view` and `transpose` together. An inadvertent `reshape` may not preserve your dimensions (axes). Using `view` to change shape protects against this: it throws an error if the dimensions being manipulated are not contiguous. 
* Shape *Annotations* vs *Assertions*. Shape annotations (`x: (B,T,D)`) ease shape recall during coding. Shape assertions (`assert x.shape === (B,T,D)`) enable catching inadvertent shape bugs at runtime. Pick either or both to work with.


## References

* Blog [article](https://medium.com/@ekshakhs/introducing-tensor-shape-annotation-library-tsalib-963b5b13c35b) introducing TSA.
* A [proposal](https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY/edit#heading=h.rkj7d39awayl) for designing a tensor library with named dimensions from ground-up. The TSA library takes care of some use cases, without requiring any change in the tensor libraries.
* Pytorch Issue on Names Axes [here](https://github.com/pytorch/pytorch/issues/4164).
* Using [einsum](http://ajcr.net/Basic-guide-to-einsum/) for tensor operations improves productivity and code readability. [blog](https://rockt.github.io/2018/04/30/einsum)
* The [Tile](https://vertexai-plaidml.readthedocs-hosted.com/en/latest/writing_tile_code.html) DSL uses indices ranging over dimension variables to write compact, library-independent tensor operations.
* The [datashape](https://datashape.readthedocs.io/en/latest/) library introduces a generic type system and grammar for structure data. `tsalib` focuses on shapes of homogeneous tensor data types only, with arithmetic support.
* The [xarray](https://github.com/pydata/xarray) library.
* The [einops](https://github.com/arogozhnikov/einops) library.
* The [namedtensor](http://nlp.seas.harvard.edu/NamedTensor) library.

## Author

 * Nishant Sinha, [OffNote Labs](http://offnote.co) (nishant@offnote.co, @[medium](https://medium.com/@ekshakhs), @[twitter](https://twitter.com/ekshakhs))

## Change Log
The library is in its early phases. Contributions/feedback welcome!

* [4 Feb 2019] Added fully annotated and adapted BERT [model](models/bert). More illustrative pytorch and tensorflow snippets.
* [31 Jan 2019] Added `alignto` operator.
* [18 Dec 2018] Added the `join` operator. `warp` takes a list of (shorthand) transformations.
* [28- Nov 2018] Added `get_dim_vars` to lookup dim vars declared earlier. Shorthand notation docs.
* [21 Nov 2018] Added documentation [notebook](notebooks/tsalib.ipynb). 
* [18 Nov 2018] Support for `warp`, `reduce_dims`. Backend modules for `numpy`, `tensorflow` and `torch` added.
* [9 Nov 2018] Support for shorthand notation in view/permute/expand transforms.
* [9 Nov 2018] Support for using TSA in assertions and tensor constructors (cast to integers).
* [25 Oct 2018] Initial Release


