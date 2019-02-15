from .tsn import tsn_to_str_list, tsn_to_tuple
from .ts import is_dummy

def get_nth_symbol(n, first=97):
    #48 (0), 65(A), 97(a)
    #first = 945 #alpha
    return chr(first+n)

def get_lowercase_symbols(n, except_char=None):
    symbols = [chr(97+i) for i in range(26)]
    if except_char: symbols.remove(except_char)
    return symbols[:n]

def unify_tuples (t1, t2):
    '''
    t1 and t2 can unifiable if 
    - lengths same
    - at each i, t1[i] and t2[i] are same or one of them is a dummy
    returns map from dummy symbols to actua dim vars
    '''
    assert isinstance(t1, tuple) and isinstance(t2, tuple)
    assert len(t1) == len(t2), f'Cannot match {t1} and {t2} of different lengths'

    dummy2dv = {}
    for t1i, t2i in zip(t1, t2):
        if t1i == t2i: 
            #res.append(t1i)
            continue
        assert not (is_dummy(t1i) and is_dummy(t2i)), f'both dummies {t1i}, {t2i}'

        if is_dummy(t1i): dummy2dv[t1i] = t2i #res.append(t2i)
        elif is_dummy(t2i): dummy2dv[t2i] = t1i #res.append(t1i)
        else:
            assert False, f"Cannot unify {t1i} and {t2i}"

    return list(dummy2dv.items())

def int_shape(*s):
    if len(s) == 1: 
        assert isinstance(s, (tuple,list))
        s = s[0]
    else: s = tuple(s)
    return tuple([int(d) for d in s])

def select(x, dv_dict, squeeze=False):
    '''
    Index using dimension shorthands
    
    x: (t, 'bcd') -- tensor, shape tuple (can be indexed in numpy notation : x[:,0,:])
    dv_dict: {'b': 0, 'c': 5} 
    squeeze: [True, False] or a tsn list ('b,c') of dims to be squeezed
    '''
    assert not squeeze, 'not implemented'

    assert isinstance(tuple), 'The first argument should be a tuple of (vector, shape)'
    xv, xs = x
    shape, is_seq = tsn_to_str_list(xs)
    if not is_seq:
        raise NotImplementedError(f"get from shape {xs} not implemented")

    colon = slice(None)
    slice_tuple = [colon] * len(shape)
    for pos, sh in shape:
        if sh in dv_dict:
            slice_tuple[pos] = dv_dict[sh]

    y = x[slice_tuple]
    return y

def size_assert(x_size, sa, dims=None):
    '''
    x_size: integer tuple
    sa: TSA
    dims: None or Sequence[int], e.g., [0,1]
    Check if size of tensor x matches TSA `sa` along `dims` axes
    '''
    x_size, sa = tuple(x_size), tuple(sa)
    if dims is not None:
        assert isinstance(dims, (list, tuple))
        x_size = [x_size[d] for d in dims]
        sa = [sa[d] for d in dims]

    if x_size != sa:
        print(f'Size mismatch: size = {x_size}, expected: {sa}')
        assert False


def reduce_dims (tfm):
    '''
    tfm: str, 'btd->b'
    '''
    src, to = tfm.split('->')
    src = tsn_to_tuple(src.strip())
    to = tsn_to_tuple(to.strip())

    assert isinstance(src, tuple)
    assert isinstance(to, tuple)

    drops = []
    #check src includes all dims in to
    for i, d in enumerate(src):
        if d not in to:
            drops.append(i)

    return tuple(drops)

