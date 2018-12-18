from tsalib.ts import TS, dim_var, dummy_dvar
from sympy import sympify
from .utils import _sexprs_to_ts, _to_tuple, check_int_tuple, resolve_to_int_tuple



def _view_transform (src, to, in_shape):
    '''
    View Transform
    :src is the current view of the tensor
    :to is the target view, may contain expressions over named dim variables
    :in_shape is the shape (list/tuple) of the source tensor 
    :returns the new size of the tensor after view transformation
    '''
    check_int_tuple(in_shape)

    src = _to_tuple(src)
    if (len(src) != len(in_shape)):
        print (f'{src}, {in_shape}')
        raise ValueError("Source TS does not match input tensor's shape")
    to = _to_tuple(to)
    #print (src, to)
    assert isinstance(in_shape, (list, tuple))

    #sub_map = [(d.e, Symbol(f'{i}')) for i, d in enumerate(src)]
    sub_map = [(d.exp, in_shape[i]) for i, d in enumerate(src)]
    out_shape = tuple([t.exp.subs(sub_map) if isinstance(t, TS) else int(t) for t in to])

    out_shape = resolve_to_int_tuple(out_shape)
    return out_shape

def view_transform (tfm, in_shape):
    '''
    View transform
    :tfm is the shorthand representation of the transform ('btd -> b,t*2,d//2')
    '''
    l, r = tfm.split('->')
    return _view_transform(l.strip(), r.strip(), in_shape)


def _permute_transform(src, to):
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
    perm_indices = resolve_to_int_tuple(perm_indices)

    return perm_indices


def permute_transform (tfm):
    '''
    Permute transform
    :tfm is the shorthand representation of the transform (',t,d -> ,d,t')
    '''
    l, r = tfm.split('->')
    return _permute_transform(l.strip(), r.strip())

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

    res = tuple(res)
    res = resolve_to_int_tuple(res)
    return res

def reduce_dims (tfm):
    '''
    tfm: 'btd->b'
    '''
    src, to = tfm.split('->')
    src = _to_tuple(src.strip())
    to = _to_tuple(to.strip())

    drops = []
    #check src includes all dims in to
    for i, d in enumerate(src):
        if d not in to:
            drops.append(i)

    return tuple(drops)


from .backend import get_backend_by_name, get_backend_for_tensor

def join (tlist, fmt, backend=None):
    assert isinstance(tlist, list), "Can only group a list of tensors"
    assert len(tlist) > 1, "Can only group more than one tensors"

    if backend is not None:
        be = get_backend_by_name(backend)
    else:
        be = get_backend_for_tensor(tlist[0])

    if '->' in fmt: #l, r style transformation
        raise NotImplementedError

    out_shape = fmt.strip().split(',')
    if '^' in out_shape:
        pos = out_shape.index('^')
        return be.stack(tlist, axis=pos)
    else:
        if '*' not in out_shape:
            assert pos != '-1', 'Group specification does not contain "^" or "*"'

        pos = out_shape.index('*')
        return be.concat(tlist, axis=pos)


def tfm_seq_decompose (tfms, tfm_names):
    '''
    Decompose a multi-step transform into basic (view, permute, expand) transforms
    tfms  'btd -> b,t,2,d//2 -> b,2,t,d//2'
    tfm_names 'vp' [first view, then permute transform]
    '''
    tfm_symbols = list(tfm_names) # ['v', 't']
    tfm_symbols_no_c = list(tfm_names.replace('c',''))
    tfm_list = [] # (trf symbol, trf_lhs, trf_rhs)

    if isinstance(tfms, str):

        shapes = [t.strip() for t in tfms.split('->')]
        assert len(shapes) >= 2, "Specify at least one transform, e.g., btd->dtb"
        assert len(tfm_symbols_no_c) == (len(shapes) - 1), "Num of transform descriptions and symbols do not match"

        shapes = [_to_tuple(s) for s in shapes]
        #for i, (l, r) in enumerate(zip(shapes[:-1], shapes[1:])):
        #    tfm_list.append((tfm_symbols[i], l, r) )

        curr_shape_pos = 0 #count current tfm's position (handle implicit contiguous)
        for sym in tfm_symbols:
            #contiguous transform
            if sym == 'c': 
                tfm_list.append((sym, None, None))
            else:
                l, r = shapes[curr_shape_pos: curr_shape_pos+2]
                tfm_list.append((sym, l, r))
                curr_shape_pos += 1

    elif isinstance(tfms, list):
        assert len(tfms) == len(tfm_symbols_no_c), "Num transformations {0} != transform symbols {1}".format(len(tfms),len(tfm_symbols_no_c))
        assert len(tfms) > 0, "No transformations given. Specify at least one transformation"
        curr_pos = 0 #count current tfm's position (handle implicit contiguous)
        for sym in tfm_symbols:
            if sym == 'c':   #contiguous transform
                tfm_list.append((sym, None, None))
            else:
                l, r = [t.strip() for t in tfms[curr_pos].split('->')]
                tfm_list.append((sym, l, r))
                curr_pos += 1

    else:
        assert False, "warp: wrong format for transforms. Specify either a string or a list."

    return tfm_list



def warp (x, tfms, tfm_names, backend=None, debug=False):
    '''
    Perform a multi-step transform on the tensor x
    tfms  'btd -> b,t,2,d//2 -> b,2,t,d//2 -> b,2,t,^n,d//2'
    tfm_names 'vp' [first (v)iew, then (p)ermute transform]
    backend    either a string('numpy', 'tf', 'torch') or the corresponding backend.<class>
    '''
    if backend is not None:
        be = get_backend_by_name(backend)
    else:
        be = get_backend_for_tensor(x)

    tfm_list = tfm_seq_decompose(tfms, tfm_names)

    #print (f'tfm list {tfm_list}')

    ret = x
    for sym, l, r in tfm_list:
        if debug:
            print(f'*** processing transform.. {sym}\n {l} -> {r}')
        if sym == 'v' or sym == 'r': #view transform
            new_shape = _view_transform(l, r, ret.shape)
            ret = be.view(x, new_shape)
        elif sym == 'p' or sym == 't':
            perm_indices = _permute_transform(l, r)
            ret = be.transpose(ret, perm_indices)
        elif sym == 'e':
            expand_shape = expand_transform(l, r)
            ret = be.expand(ret, expand_shape)
        elif sym == 'c': 
            ret = be.contiguous(ret)
        else:
            assert False, f'Invalid transform symbol {sym}'
        if debug:
            print (f'after transform shape is: {be.shape(ret)}')
    return ret











