from .utils import int_shape

class ABackend:
    name = None

    def shape(self, x): raise NotImplementedError
    def contiguous(self, x): raise NotImplementedError
    def view(self, x, shape): raise NotImplementedError
    def transpose(self, x, dims): raise NotImplementedError
    def expand(self, x, mul_shape): raise NotImplementedError
    def stack(self, x, mul_shape): raise NotImplementedError
    def concat(self, x, mul_shape): raise NotImplementedError
    def einsum(self, eqn, args): raise NotImplementedError


class Numpy(ABackend):
    name = 'numpy'

    def __init__(self):
        import numpy
        self.np = numpy
    def shape(self, x): return x.shape
    def contiguous(self, x): self.np.ascontiguousarray(x)
    def view(self, x, shape): 
        #print (type(x), x.shape, shape)
        return x.reshape(shape)
    def transpose(self, x, dims): return x.transpose(dims)
    def expand(self, x, mul_shape): return x.tile(mul_shape)
    def stack(self, xlist, axis): return self.np.stack(xlist, axis=axis)
    def concat(self, xlist, axis): return self.np.concatenate(xlist, axis=axis)
    def einsum(self, eqn, args): return self.np.einsum(eqn, *args)

class PyTorch(ABackend):
    name = 'pytorch'
    def __init__(self):
        import torch
        self.torch = torch
    def shape(self, x): return x.size()
    def contiguous(self, x): return x.contiguous()
    def view(self, x, shape): return x.view(shape)
    def transpose(self, x, dims): return x.permute(dims)
    def expand(self, x, mul_shape): return x.expand(mul_shape)
    def stack(self, xlist, axis): return self.torch.stack(xlist, dim=axis)
    def concat(self, xlist, axis): return self.torch.cat(xlist, dim=axis)
    def einsum(self, eqn, args): return self.torch.einsum(eqn, args)


def get_tf_shape(tf, tensor):
  """Returns a list of the shape of tensor, preferring static dimensions.
  (inspired by get_shape_list in BERT code)

  Args:
    tensor: A tf.Tensor object to find the shape of.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  shape = tuple(tensor.shape.as_list())

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

class TF(ABackend):
    name = 'tensorflow'
    def __init__(self):
        import tensorflow
        self.tf = tensorflow

    def shape(self, x):
        if self.tf.executing_eagerly():
            return int_shape(x.shape)
        else:
            #return tuple(self.tf.unstack(self.tf.shape(x)))
            return get_tf_shape(self.tf, x)

    def contiguous(self, x): return x

    def view(self, x, shape): return self.tf.reshape(x, shape)
    def transpose(self, x, dims): return self.tf.transpose(x, dims)
    def expand(self, x, mul_shape): return self.tf.tile(x, mul_shape)
    def stack(self, xlist, axis): return self.tf.stack(xlist, axis=axis)
    def concat(self, xlist, axis): return self.tf.concat(xlist, axis=axis)
    def einsum(self, eqn, args): return self.tf.einsum(eqn, args)


becache  = {}
def from_cache(C):
    s = str(C)
    if s not in becache:
        becache[s] = C()
    return becache[s]

def get_str_type(x):
    #print (x)
    if isinstance(x, (tuple,list)):
        if len(x) == 0: return ''
        x = x[0]
        
    t = str(type(x))
    return t

def get_tensor_lib(x):
    t = get_str_type(x)

    if 'numpy.' in t: ret = Numpy
    elif 'torch.' in t: ret = PyTorch
    elif 'tensorflow.' in t: ret = TF
    else: ret = None

    return ret

def is_tensor(x):
    t = get_str_type(x)

    if 'numpy.' in t: ret = ('numpy.ndarray' in t)
    elif 'torch.' in t: ret = ('torch.Tensor' in t)
    elif 'tensorflow.' in t: ret = ('ops.Tensor' in t)
    else: ret = False

    return ret

def get_backend_for_tensor(x):
    '''
    get backend for tensor x
    '''
    tlib = get_tensor_lib(x)

    if tlib is None:
        raise NotImplementedError(f'Unsupported tensor type {type(x)}. Contributions welcome.')
    
    ret = from_cache(tlib)

    return ret

def get_backend_by_name (b):
    if isinstance(b, (Numpy, TF, PyTorch)): return b
    assert isinstance(b, str)
    bemap = {
        'numpy': Numpy,
        'np': Numpy,
        'torch': PyTorch,
        'pytorch': PyTorch,
        'tensorflow': TF,
        'tf': TF
    }
    if b in bemap: return from_cache(bemap[b])
    else:
        raise NotImplementedError(f'Unsupported backend {b}. Contributions welcome.')




