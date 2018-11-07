from sympy import symbols, Integer
from sympy import Symbol, nan, simplify
import re

def arith_op (op, s1, s2):
    assert isinstance(s1, TS)
    s2 = TS(s2)

    s1e = s1.exp
    s2e = s2.exp

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

    return TS(se)

class DimVar:
    decls = {} #caches all dim var declarations
    parse_regexp = r'(\w+)(?:\((\w)\))?(?::(\d+))?' #Height(h)?(:300)?

    def __init__ (self, decl, strict):
        assert isinstance(decl, str)
        decl = decl.strip()

        m = re.search(DimVar.parse_regexp, decl)
        name, sname, val = m.groups()
        #print (m.groups())

        self._name = name
        self._sname = sname if sname is not None else name
        self._val = int(val) if val is not None else nan
        
        self._e = Symbol(self._sname)
        if strict and self._e in DimVar.decls:
            raise ValueError(f'DimVar {self._sname} already declared.')
       
        DimVar.decls[self._e] = self

    @property
    def exp(self): return self._e

    @property
    def len(self): return self._val

    @property
    def name(self):
        ret = f'{self._name}'
        if self._name != self._sname: ret += f'({self._sname})'
        return ret

    @staticmethod
    def check_decl(sname):
        return Symbol(sname) in DimVar.decls

    @staticmethod
    def eval(e):
        sub_map = [(e, dv._val) for e, dv in DimVar.decls.items()]
        return e.subs(sub_map)

    @staticmethod
    def eval_name(e):
        sub_map = [(e, dv.name) for e, dv in DimVar.decls.items()]
        return str(e.subs(sub_map))

class TS:
    '''
    The Tensor Shape Expression Class
    '''
    DEFAULT_VALUE = 1

    def __init__(self, t, is_dvar=False):
        self._e = None
        self.is_dvar = is_dvar # a basic dimension var

        if isinstance(t, int):
            self._e = Integer(t)
            self._val = t
        elif isinstance(t, DimVar):
            self._e, self._val, self.is_dvar = t.exp, t.len, True
        elif isinstance(t, TS):
            self._e, self._val, self.is_dvar = t._e, t._val, t.is_dvar
        else:
            #print (f'test expr: {v} {repr(type(v))}')
            self._e = t
            self._val = DimVar.eval(t)

    @property
    def exp(self): return self._e
    @property
    def len(self): 
        return self._val if (self._val != nan) else None

    def __int__(self): 
        if self._val != nan:
            return self._val
        else: return TS.DEFAULT_VALUE

    def __add__(self, n): return arith_op('add', self, n)
    def __mul__(self, n): return arith_op('mul', self, n)
    def __floordiv__(self, n): return arith_op('floordiv', self, n)
    #truediv: '/' provided for convenience; prefer using '//'
    def __truediv__(self, n): return arith_op('truediv', self, n)

    def __eq__(self, d):
        #TODO: being conservative here, may need to generalize this
        if not isinstance(d, TS): return False
        return self._e == d._e    

    def __hash__(self):
        return hash(self._e)

    def __repr__(self):
        s = DimVar.eval_name(self._e)
        if self._val != nan:
            s += f':{self._val}'
        return s


def dim_var (name, strict=True):
    '''
    Declare a single dimension variable
    '''
    d = DimVar(name, strict=strict)
    return TS(d)

def dim_vars(names, strict=True):
    '''
    Declare multiple dimension variables in one go
    '''
    names = names.strip().split(' ')
    tss = [dim_var(name, strict=strict) for name in names]

    if len(names) == 1: return tss[0]
    else: return tss

def declare_common_dim_vars ():
    B, V, D, Dh = dim_vars('Batch Vocab EmbedDim HiddenDim')
    C, Ci, Co = dim_vars('Channels InChannels OutChannels')
    T, Te, Td = dim_vars('Time EncoderTime DecoderTime')

    return B, D, V, Dh, T, Te, Td, C, Ci, Co
