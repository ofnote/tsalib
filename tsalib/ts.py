from sympy import symbols, Integer
from sympy import Symbol, nan, simplify


def arith_op (op, s1, s2):
    assert isinstance(s1, TS)
    s2 = TS(s2)

    valmap = {}
    if s1.is_dvar: valmap[s1.e] = s1.val
    if s2.is_dvar: valmap[s2.e] = s2.val

    s1e = s1.e
    s2e = s2.e

    #print (f'arith_op: {op} {s1} {s2}')
    if op == 'add':
        se = s1e + s2e
    elif op == 'mul':
        se = s1e * s2e
    elif op == 'truediv':
        se = s1e / s2e  
    elif op == 'floordiv':
        se = s1e // s2e  
    else:
        raise NotImplementedError(f'{op}')

    s_val = se.subs(valmap)

    return TS(se, s_val)


class TS:
    '''
    The Tensor Shape Expression Class
    '''
    def __init__(self, v, value=nan):
        self.val = value
        self.e = None
        self.is_dvar = False # a basic dimension var

        if isinstance(v, str):
            names = v.strip().split(' ')
            assert len(names) == 1  #only allow single token names
            self.e = Symbol(v)
            self.is_dvar = True

        elif isinstance(v, int):
            self.e = Integer(v)
        elif isinstance(v, TS):
            self.e, self.val, self.is_dvar = v.e, v.val, v.is_dvar
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
        if self.val != nan:
            s += f':{self.val}'
        return s


def dim_var (name):
    '''
    Declare a single dimension variable
    '''
    val = nan
    if ':' in name:
        name, val = name.strip().split(':')
        val = int(val)
    return TS(name, val)

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
