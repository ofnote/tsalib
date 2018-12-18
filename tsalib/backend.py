from .utils import to_int_tuple

class ABackend:
    name = None

    def shape(self, x): raise NotImplementedError
    def contiguous(self, x): raise NotImplementedError
    def view(self, x, shape): raise NotImplementedError
    def transpose(self, x, dims): raise NotImplementedError
    def expand(self, x, mul_shape): raise NotImplementedError
    def stack(self, x, mul_shape): raise NotImplementedError
    def concat(self, x, mul_shape): raise NotImplementedError


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


class TF(ABackend):
    name = 'tensorflow'
    def __init__(self):
        import tensorflow
        self.tf = tensorflow

    def shape(self, x):
        if self.tf.executing_eagerly():
            return to_int_tuple(x.shape)
        else:
            return tuple(self.tf.unstack(self.tf.shape(x)))
    def contiguous(self, x): return x

    def view(self, x, shape): return self.tf.reshape(x, shape)
    def transpose(self, x, dims): return self.tf.transpose(x, dims)
    def expand(self, x, mul_shape): return self.tf.tile(x, mul_shape)
    def stack(self, xlist, axis): return self.tf.stack(xlist, axis=axis)
    def concat(self, xlist, axis): return self.tf.concat(xlist, axis=axis)


becache  = {}
def from_cache(C):
    s = str(C)
    if s not in becache:
        becache[s] = C()
    return becache[s]

def get_backend_for_tensor(x):
    '''
    get backend for tensor x
    '''
    t = str(type(x))

    if 'numpy.' in t: ret = from_cache(Numpy)
    elif 'torch.' in t: ret = from_cache(PyTorch)
    elif 'tensorflow.' in t: ret = from_cache(TF)
    else: 
        raise NotImplementedError(f'Unable to handle tensor of type {t}.')

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




