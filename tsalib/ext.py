from tsalib.ts import TS, dim_var, dummy_dvar
from sympy import sympify


def _sexpr_to_ts (e, dummy_idx=-1, strict=False):
    '''
    A single string expression (sexpr) to Tensor Shape expressions (ts)
    Converts shorthand dummy/empty placeholders to dummy TSs
    '''
    if isinstance(e, TS):  
        t = e
    else: 
        assert isinstance(e, str)
        if e == '' or e =='_':
            t = dummy_dvar(dummy_idx)
            dummy_idx += 1
        else: 
            #TODO: strict: check if all dim vars in e are previously declared?
            t = TS(sympify(e))

    return t, dummy_idx

def _sexprs_to_ts(exprs, strict=False):
    '''
    String expressions (sexprs) to Tensor Shape expressions (ts)
    Converts shorthand dummy/empty placeholders to dummy TSs
    Returns a tuple of TSs
    '''
    dummy_idx = 0
    res = []
    for e in exprs:
        t, dummy_idx = _sexpr_to_ts(e, dummy_idx, strict)
        res.append(t)

    #print (exprs, res)
    return tuple(res)


def _to_tuple (ss):
    '''
    :ss is shape string, e.g., ('btd') or ('b,t,d*2')
    '''
    if isinstance(ss, (list, tuple)):
        for s in ss: assert isinstance(s, (TS,int))
        return tuple(ss)

    elif isinstance(ss, str):
        if ',' in ss: exprs = ss.strip().split(',') #'b,t,d*2' -> ['b', 't', 'd*2']
        else: exprs = list(ss)              # 'btd' -> ['b','t','d']

        exprs = _sexprs_to_ts(exprs)
        return tuple(exprs)

    else:
        raise ValueError('Unknown type of ss')

def check_int_tuple(s):
    #print(f'int tuple? {s}')
    for d in s:
        try: d = int(d)
        except:
            raise ValueError(f'Unable to resolve expression {d}')

def view_transform (src, to, in_shape):
    '''
    View Transform
    :src is the current view of the tensor
    :to is the target view, may contain expressions over named dim variables
    :in_shape is the shape (list/tuple) of the source tensor 
    :returns the new size of the tensor after view transformation
    '''
    check_int_tuple(in_shape)

    src = _to_tuple(src)
    assert (len(src) == len(in_shape)), "Source TS does not match input tensor's shape"
    to = _to_tuple(to)
    #print (src, to)
    assert isinstance(in_shape, (list, tuple))

    #sub_map = [(d.e, Symbol(f'{i}')) for i, d in enumerate(src)]
    sub_map = [(d.exp, in_shape[i]) for i, d in enumerate(src)]
    out_shape = tuple([t.exp.subs(sub_map) if isinstance(t, TS) else int(t) for t in to])

    check_int_tuple(out_shape)
    return out_shape


def to_shape (src):
    '''
    src: 'b,t,h*d'
    Lookup each shorthand in cache. 
    returns: (B, T, H*D)
    '''
    return NotImplemented

def permute_transform(src, to):
    '''
    Permute Transform
    :src is the current dimension arrangement, list of named dim variables
    :to is the target dimension arragement, list of named dim variables
    :returns the index tuple for the permutation
    '''
    src = _to_tuple(src)
    to = _to_tuple(to)

    assert len(src) == len(to), "Source and Target shapes for permutation are not same"

    sub_map = [(d.exp, i) for i, d in enumerate(src)]
    perm_indices = tuple([t.exp.subs(sub_map) for t in to])
    check_int_tuple(perm_indices)

    return perm_indices

def _expansions_as_dict(expansions):
    if isinstance(expansions, list): #[(T, T*5), (D, D*4)]
        res = expansions
    else:
        assert isinstance(expansions, str) #'k->k*5,t->t*10'
        expansions = expansions.strip().split(',')
        res = []
        for ex in expansions:
            t = _sexprs_to_ts(ex.strip().split('->'))
            #print(t)
            res.append(t)
            #print (res)

    exp_map = {l: r for (l, r) in res}
    return exp_map


def expand_transform (src, expansions, in_shape):
    '''
    :src        (B, T, D) = (10, 20, 300)
    :expansions [(T, T*5), (D, D*4)]
    :returns the expansion shape tuple
    '''
    src = _to_tuple(src)
    exp_map = _expansions_as_dict(expansions)
    #exp_map = {_sexpr_to_ts(s)[0]: _sexpr_to_ts(t)[0] for (s, t) in (expansions)}
    sub_map = [(d.exp, in_shape[i]) for i, d in enumerate(src)] # (B, 10), (T, 20), (D, 300)

    #print (expansions, exp_map)

    res = []
    for k in src:
        if k not in exp_map: res.append(-1) #keep dim shape same
        else:
            v = exp_map[k]
            assert isinstance(v, TS)
            res.append(v.exp.subs(sub_map)) #(T*5) -> 100, (D*4) -> 1200

    res =tuple(res)
    check_int_tuple(res)
    return res







