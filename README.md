# Tensor Shape Annotation Library (tsalib)

Writing programs which manipulate tensors (e.g., using `numpy`, `pytorch`, `tensorflow`, ..) requires you to carefully keep track of shapes of tensor variables. Carrying around the shapes in your head gets increasingly hard as programs become more complex, e.g., when creating a new `RNN` cell or designing a new kind of `attention` mechanism or trying to do a surgery of non-trivial pre-trained architectures (`resnet101`, `densenet`). There is no principled way of shape tracking inside code -- most developers resort to writing adhoc comments embedded in code to keep track of tensor shapes.

The `tsalib` library comes to your rescue here. It allows you to label tensor variables with their shapes directly in the code, as *first-class* type annotations. Shape annotations turn out to be useful in many ways. They help you quickly cross-check the variable shapes when writing new transformations or modifying existing modules. Moreover, the annotations serve as useful documentation to guide others trying to understand or extend your module.

* Because shapes can be dynamic, you can write `symbolic` shape expressions over named dimension variables, e.g., 

    `v : (B, C, H, W) = torch.randn(batch_size, channels, h, w)`

    Here `B`, `C`, `H`, `W` are pre-defined named dimension variables. It is easy to define new named dimensions customized to your architecture. Of course, use constant values if one or more dimensions remain fixed for all inputs.

    `v : (B, 64, H, W) = torch.randn(batch_size, 64, h, w)`

* Arithmetic over shapes is supported

    `x : (B, C * 2, H/2, W/2) = torch.nn.conv2D(ch_in, ch_in*2, ...)(v)`

* Works seamlessly with arbitrary tensor libraries:  `numpy`, `pytorch`, `tensorflow`, `mxnet`, etc. 

## Getting Started

See `examples/test.py`.


## Dependencies

Python >= 3.6. Allows harmless type annotations for variables. These annotations do not affect the program performance in any way. 

`sympy`. A library for building symbolic expressions in Python.

