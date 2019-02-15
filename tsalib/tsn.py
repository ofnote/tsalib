'''
Utilities for parsing, transforming tensor shorthand notation (TSN)
'''

from .ts import DimVar, DimExpr, dummy_dvar, TupleSeq
from sympy import sympify, Symbol
import re

def _sexpr_to_ts (e, dummy_idx=0, strict=False, num_to_sym=False):
    '''
    A single string expression (sexpr) to Tensor Shape expressions (ts)
    Converts shorthand dummy/empty placeholders to dummy TSs
    '''
    if isinstance(e, DimExpr):  
        t = e
    else: 
        assert isinstance(e, str)
        if e.isdigit() and num_to_sym: e = '_' #convert to dummy var
        if e == '' or e =='_':  
            t = dummy_dvar(dummy_idx)
            dummy_idx += 1
        elif e == '^':
            #TODO: better way to handle '^' ?
            t = DimExpr(Symbol(e))
        else: 
            #TODO: strict: check if all dim vars in e are previously declared?
            t = DimExpr(sympify(e))

    return t, dummy_idx

def _sexprs_to_ts(exprs, strict=False, num_to_sym=False):
    '''
    String expressions (sexprs) to Tensor Shape expressions (ts)
    Converts shorthand dummy/empty placeholders to dummy TSs
    Returns a tuple of TSs
    '''
    dummy_idx = 0
    res = []
    for e in exprs:
        t, dummy_idx = _sexpr_to_ts(e, dummy_idx=dummy_idx, strict=strict, num_to_sym=num_to_sym)
        res.append(t)

    #print (exprs, res)
    return tuple(res)


seq_re = r'\((.+)\)\*'

def tsn_to_str_list(ss: str):
    # 'btd' -> ['b','t','d']

    #remove all whitespace characters
    ss = re.sub(r'\s+', '', ss)

    #check if shape corresponds to a sequence        
    is_seq = False
    m = re.search(seq_re, ss)
    if m is not None:  # ss = '(b,t,d)*'
        ss = m.groups()[0]  # ss = 'b,t,d'
        #print (f'groups: {m.groups()}') 
        is_seq = True

    if ',' in ss: exprs = ss.strip().split(',') #'b,t,d*2' -> ['b', 't', 'd*2']
    else: exprs = list(ss)  

    return exprs, is_seq 

def tsn_to_tuple (ss, num_to_sym=False):
    '''
    :ss is shape string, e.g., 'btd' or 'b,t,d*2' or '(btd)*'
    : num_to_sym : converts numeric values in tsn to anonymous symbols ('_')
    :returns the shape representation in tuple/TupleSeq form
    '''
    if isinstance(ss, (list, tuple)):
        for s in ss: assert isinstance(s, (DimExpr,int))
        return tuple(ss)
    elif isinstance(ss, TupleSeq):
        return ss
    elif isinstance(ss, str):
        exprs, is_seq = tsn_to_str_list(ss)  # 'btd' -> 'b', 't', 'd'
        exprs = _sexprs_to_ts(exprs, num_to_sym=num_to_sym)
        for e in exprs:
            assert isinstance(e, DimExpr)

        exprs = tuple(exprs)

        if is_seq:
            exprs = TupleSeq(exprs)

        return exprs

    else:
        raise ValueError('Unknown type of ss')

def check_int_tuple(s):
    #print(f'int tuple? {s}')
    for d in s:
        try: d = int(d)
        except:
            raise ValueError(f'Unable to resolve expression {d}')
def is_int_tuple(s):
    ret = all([isinstance(x, int) for x in s])
    return ret

def resolve_to_int_tuple(s):
    '''
    resolve non-int elements by casting to int or looking up their DimVar values
    '''
    res = []
    for d in s:
        try: 
            #print (type(d), d)
            d = int(d)
            res.append(d)
        except:
            if isinstance(d, DimExpr):
                e = d.exp
            else:
                e = d
                #raise ValueError(f'Unknown item {d}: {type(d)}')

            r = DimVar.eval(e)
            #print('r is ', r)
            try: 
                r = int(r)
                res.append(r)
            except:
                raise ValueError(f'Unable to resolve {d}')

    return tuple(res)



def tsn_to_shape (tsn):
    '''
    tsn: 'b,t,h*d'
    Lookup each shorthand in cache. 
    returns: (B, T, H*D)
    '''
    assert isinstance(tsn, str)
    return tsn_to_tuple(tsn)