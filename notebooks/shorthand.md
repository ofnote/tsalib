# Tensor Shorthand Notation (TSN)

`tsalib` allows using shorthand names to refer to dimension/axes in variable annotations and shape transformations. 

## Symbols for Naming Dimensions

Declare dimension variables `B` and `C` along with their shorthands `b` and `c` respectively. 

```python
B, C, D = dim_vars('Batch(b):10 C(c):3 D(d):100')
```
Then, use shorthands `b`, `c`, `d`, throughout the code, in named shape annotations or `warp` shape transformations. 

Named shapes also use a few special symbols:
- `*` to denote a sequence of shapes, e.g., `(b,t,d)*` denotes a sequence of tensors, each with shape `(b,t,d)`.
- '^' to denote a new (anonymous) dimension, e.g., `(b,t,^,d)` denotes a tensor with a new dimension at 3rd position. Used when expanding dimensions of tensors.

## Named Shapes

* In the simplest case, a named shape is a tuple of dimension names, without spaces, e.g., `bcd`. Here, each character is interpreted as an unique dimension.
```python
_ = permute_transform('bcd -> bdc')
```

* If a named shape both names and expressions, use comma format: `b,t,c * d`. In this format, a comma must follow each dimension (except the last one).

* Skip dimensions irrelevant to a transformation (*comma* or *_*) : 
    - write 'bcd' as 'b,,d' or 'b_d' (make dimension `c` anonymous)
    - ```permute_transform('_cd -> _dc')```

* A named shape can include constants and expressions over dimension names, e.g., `b,1,c*d`.

## Resolving Shorthands, Shape Transformations

In shape transformations of form `lhs -> rhs` where `lhs` and `rhs` are named shapes, shorthand characters are resolved to pre-declared dimension variables. We recommend that all shorthands are declared upfront.



## Warp Transformation

In the `warp` transformation, we can also specify shorthand names for shape transformations. The following shorthands are supported:
- `v`  or `r` for view/reshape transformations
- `p`  or `t` for permute/transpose transformations
- `j` for join transformations (stack/concatenate)
- `e` for expand transformation
- `c` for contiguous transformation



