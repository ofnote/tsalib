from sympy import symbols, Integer
from sympy import Symbol, nan, simplify
import re

def arith_op (op, s1, s2):
    assert isinstance(s1, DimExpr)
    s2 = DimExpr(s2)

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

    return DimExpr(se)

class TupleSeq:
    def __init__(self, s):
        self.s = s
    def item(self): return self.s

class DimVar:
    decls = {} #caches all dim var declarations
    parse_regexp = r'(\w+)(?:\((\w+)\))?(?::(\d+))?' #Height(h)?(:300)?

    def __init__ (self, decl, exists_ok, cache):
        '''
        :decl: declaration string of variable ('Batch(b):20')
        :exists_ok: if declared earlier, nop
        :cache: store in `decls` cache
        '''
        assert isinstance(decl, str)
        decl = decl.strip()

        m = re.search(DimVar.parse_regexp, decl)
        name, sname, val = m.groups()
        #print (m.groups())

        self._name = name
        self._sname = sname if sname is not None else name
        self._val = int(val) if val is not None else nan
        
        self._e = Symbol(self._sname)
        if self._e in DimVar.decls:
            prevd = DimVar.decls[self._e]
            if not exists_ok:
                raise ValueError(f'DimVar {self._sname} already declared as {prevd._name}({self._e}). Use exists_ok=True to skip check.')

        else:
            if cache: DimVar.decls[self._e] = self

    @property
    def exp(self): return self._e

    @property
    def size(self): return self._val

    @property
    def shortname(self): return self._sname

    @property
    def name(self):
        ret = f'{self._name}'
        if self._name != self._sname: ret += f'({self._sname})'
        return ret

    @staticmethod
    def check_decl(sname):
        return Symbol(sname) in DimVar.decls

    @staticmethod
    def lookup(sname):
        #lookup by short name
        sn = Symbol(sname)
        #print (f'lookup: {sn} {len(DimVar.decls)}')
        if len(DimVar.decls) == 0: 
            assert False
        assert sn in DimVar.decls, f'DimVar {sn} not declared'
        return DimVar.decls[sn]

    @staticmethod
    def lookup2(name):
        #lookup by (long) name
        for k, decl in DimVar.decls.items():
            #print ('** lookup2', name, decl._name)
            if decl._name == name: return decl
        assert False, f'DimVar with name {name} not declared'

    @staticmethod
    def eval(e):
        sub_map = [(e, dv._val) for e, dv in DimVar.decls.items()]
        ret = e.subs(sub_map)
        #print (e, sub_map)
        #print (f'eval: {e} -> {ret}')
        return ret

    @staticmethod
    def eval_name(e):
        sub_map = [(e, dv.shortname) for e, dv in DimVar.decls.items()]
        return str(e.subs(sub_map))

class DimExpr:
    '''
    Encapsulates the expression for a particular axis/dimension
    '''
    #DEFAULT_VALUE = 1

    def __init__(self, t, is_dvar=False):
        self._e = None
        self.is_dvar = is_dvar # a basic dimension var
        self._val = None #value of dimvar (nan if not set)

        if isinstance(t, int):
            self._e = Integer(t)
            self._val = t
        elif isinstance(t, DimVar):
            self._e, self._val, self.is_dvar = t.exp, t.size, True
        elif isinstance(t, DimExpr):
            self._e, self._val, self.is_dvar = t._e, t._val, t.is_dvar
        else:
            #print (f'test expr: {v} {repr(type(v))}')
            self._e = t
            self._val = DimVar.eval(t)
            #self._val = int(v) if v is not nan else v

    @property
    def exp(self): return self._e
    @property
    def len(self): 
        return self._val if (self._val != nan) else None

    def __int__(self): 
        #print(f'called int {self._val}')
        if self._val != nan:
            return int(self._val)
        else: 
            #return DimExpr.DEFAULT_VALUE
            raise ValueError(f'Cannot cast to integer: Default value of {self._e} not provided')
    def __index__(self): return self.__int__()

    def __add__(self, n): return arith_op('add', self, n)
    def __radd__(self, n): return self.__add__(n)
    def __mul__(self, n): return arith_op('mul', self, n)
    def __rmul__(self, n): return self.__mul__(n)

    def __floordiv__(self, n): return arith_op('floordiv', self, n)
    def __rfloordiv__(self, n): return self.__floordiv__(n)

    #truediv: '/' provided for convenience; prefer using '//'
    def __truediv__(self, n): return arith_op('truediv', self, n)
    def __rtruediv__(self, n): return self.__truediv__(n)

    def __eq__(self, d):
        #print (f'eq: {self}, {d}')
        if isinstance(d, int):
            #semantics: any integer matches nan
            if self._val == nan: return True 
            else: return self._val == d
        elif isinstance(d, DimExpr):
            res = self._e == d._e 
            #print (res)
            return res
        else:
            return False   

    def __hash__(self):
        return hash(self._e)

    def __repr__(self):
        s = DimVar.eval_name(self._e)
        if self._val != nan:
            s += f':{self._val}'
        return s


def dim_var (name, exists_ok=False, cache=True):
    '''
    Declare a single dimension variable
    '''
    d = DimVar(name, exists_ok=exists_ok, cache=cache)
    return DimExpr(d)

def dummy_dvar(pos):
    '''
    Declare a dummy dimension variable at a particular dim position. Do not cache.
    '''
    assert pos >= 0
    name = f'_dm_{pos}'
    d = dim_var(name, exists_ok=True, cache=False)
    #print (f'dummy {d}')
    return d

def is_dummy (dvar):
    return '_dm_' in str(dvar.exp)

def dim_vars_from_shape(names, shape, exists_ok=False):
    '''
    Declare dim vars corresponding to dimensions of tensor
    :names 'b t d'
    :shape (10, 30, 300)
    '''
    names = names.strip().split(' ')
    assert len(names) == len(shape), 'Number of Dimension Variables and Shape mismatch'

    tss = [dim_var(f'{name}:{shape[i]}', exists_ok=exists_ok) for i, name in enumerate(names)]
    if len(names) == 1: return tss[0]
    else: return tss


def dim_vars(names, exists_ok=False, cache=True):
    '''
    Declare multiple dimension variables in one go
    '''
    names = names.split()
    #print (repr(names))
    tss = [dim_var(name, exists_ok=exists_ok, cache=cache) for name in names]

    if len(names) == 1: return tss[0]
    else: return tss

def get_dim_vars(names):
    '''
    names: 'b c h w', separated by spaces
    '''
    names = names.strip().split(' ')
    res = [DimExpr(DimVar.lookup(name)) for name in names]
    if len(names) == 1: return res[0]
    else: return res

def get_dim_vars_by_long_name(names):
    '''
    names: 'B Channel D'
    '''
    names = names.strip().split(' ')
    res = [DimExpr(DimVar.lookup2(name)) for name in names]
    if len(names) == 1: return res[0]
    else: return res


def get_decls (): return DimVar.decls

#def update_dim_var_size ():
#avoid this function
#maybe redeclare dim var?

def declare_common_dim_vars ():
    B, V, D, Dh = dim_vars('Batch Vocab EmbedDim HiddenDim')
    C, Ci, Co = dim_vars('Channels InChannels OutChannels')
    T, Te, Td = dim_vars('Time EncoderTime DecoderTime')

    return B, D, V, Dh, T, Te, Td, C, Ci, Co
