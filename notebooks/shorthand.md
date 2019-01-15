# Tensor Shorthand Notation (TSN)

`tsalib` allows using shorthand names to refer to dimension/axes in shape transformation specifications. 

## Declare Shorthands

Declare dimension variables `B` and `C` along with their shorthands `b` and `c` respectively. 

```python
B, C, D = dim_vars('Batch(b):10 C(c):3 D(d):100')
```
In some cases, it is possible to use `b` and `c` directly in shape annotations (TSAs) without declaring them (see *valid* shape transformations below). However, we recommend that all shorthands are declared upfront.


## Shorthand Symbol Format

* In the simplest case, single character shorthands can be written as character sequences in TSAs, e.g., `bcd`. Here, each character is interpreted as an unique dimension.
```python
_ = permute_transform('bcd -> bdc')
```

* If a TSA contains both names and expressions, use comma format: `b,t,c * d`. In this format, a comma must follow each dimension (except the last one).

* Skip dimensions irrelevant to a transformation by using placeholders (*comma* or *_*) : 
    - write 'bcd' as 'b,,d' or 'b_d'
    - ```permute_transform(',c,d -> ,d,c')```

## Resolving Shorthands, Shape Transformations

In shape transformations of form `lhs -> rhs` where `lhs` and `rhs` are TSAs, shorthand characters are first resolved to pre-declared dimension variables or new variables created.

Only *valid* shape transformations can be resolved. A shape transformation is *valid* if for all dimension variables `D` in  `rhs`: either `D` is contained in `lhs` or the size of `D` has been declared earlier.

TSAs use a few special symbols:
- `*` to denote a sequence of shapes, e.g., `(b,t,d)*` denotes a sequence of tensors, each with shape `(b,t,d)`.
- '^' to denote a new (unnamed) dimension, e.g., `(b,t,^,d)` denotes a tensor with a new dimension at 3rd position. Used when expanding dimensions of tensors.


## Warp Transformation

In the `warp` transformation, we can also specify shorthand names for shape transformations. The following shorthands are supported:
- `v`  or `r` for view/reshape transformations
- `p`  or `t` for permute/transpose transformations
- `j` for join transformations (stack/concatenate)
- `e` for expand transformation
- `c` for contiguous transformation



