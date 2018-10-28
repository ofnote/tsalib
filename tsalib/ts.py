from sympy import symbols, Integer
from sympy import Symbol


def arith_op (op, s1, s2):
    assert isinstance(s1, TS)
    s2 = TS(s2)
    s1 = s1.e
    s2 = s2.e

    #print (f'arith_op: {op} {s1} {s2}')
    if op == 'add':
        res = s1 + s2
    elif op == 'mul':
        res = s1 * s2
    elif op == 'truediv':
        res = s1 / s2  
    elif op == 'floordiv':
        res = s1 // s2  
    else:
        raise NotImplementedError(f'{op}')

    return TS(res)


class TS:
    '''
    The Tensor Shape Expression Class
    '''
    def __init__(self, v):
        self.e = None
        if isinstance(v, str):
            names = v.strip().split(' ')
            assert len(names) == 1  #only allow single token names
            self.e = Symbol(v)
        elif isinstance(v, int):
            self.e = Integer(v)
        elif isinstance(v, TS):
            self.e = v.e
        else:
            #print (f'test expr: {v} {repr(type(v))}')
            #assert 'sympy' in str(type(v))
            self.e = v

    def __add__(self, n): return arith_op('add', self, n)
    def __mul__(self, n): return arith_op('mul', self, n)
    def __floordiv__(self, n): return arith_op('floordiv', self, n)
    #truediv: '/' provided for convenience
    def __truediv__(self, n): return arith_op('truediv', self, n)

    def __eq__(self, d):
        #TODO: being conservative here, may need to generalize this
        if not isinstance(d, TS): return False
        return self.e == d.e    

    def __hash__(self):
        return hash(self.e)
    def __repr__(self):
        s = str(self.e)
        return s


def dim_var (name):
    '''
    Declare a single dimension variable
    '''
    return TS(name)

def dim_vars(names):
    '''
    Declare multiple dimension variables in one go
    '''
    names = names.strip().split(' ')
    tss = [dim_var(name) for name in names]

    if len(names) == 1: return tss[0]
    else: return tss

def declare_common_dim_vars ():
    B, V, D, Dh = dim_vars('Batch Vocab EmbedDim HiddenDim')
    C, Ci, Co = dim_vars('Channels InChannels OutChannels')
    T, Te, Td = dim_vars('Time EncoderTime DecoderTime')

    return B, D, V, Dh, T, Te, Td, C, Ci, Co
