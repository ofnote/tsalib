from .tsn import tsn_to_tuple, tsn_to_str_list
from .backend import get_backend_by_name, get_backend_for_tensor
from .transforms import _view_transform, _permute_transform, _join_transform, _expand_transform
from .transforms import alignto
from .utils import get_lowercase_symbols, unify_tuples

def get_backend(backend, x):
    if backend is not None:
        be = get_backend_by_name(backend)
    else:
        be = get_backend_for_tensor(x)

    return be

def join (tlist, dims, backend=None):
    '''
    tlist: List[tensor], list of tensors
    dims: str = '..,^,...' or '..,*,...' 
    '''
    assert isinstance(tlist, list), "Can only group a list of tensors."
    assert len(tlist) > 1, "Can only group more than one tensors."

    be = get_backend(backend, tlist[0])

    if len(dims) > 1: assert ',' in dims, 'Separate dimensions by "," here.'

    out_shape = dims.strip().split(',')
    if '^' in out_shape:
        pos = out_shape.index('^')
        return be.stack(tlist, axis=pos)
    else:
        if '*' not in out_shape:
            assert pos != '-1', 'Group specification does not contain "^" or "*".'

        pos = out_shape.index('*')
        return be.concat(tlist, axis=pos)


def tsnseq2shape_pairs (tfms):
    res = [tsn_to_tuple(t.strip()) for t in tfms.split('->')]
    assert len(res) >= 2, 'At least 2 tfms required: {res}'

    shape_pairs = list(zip(res[:-1], res[1:]))
    return shape_pairs

'''
def unify_merge (res, tfms):
    if len(res) == 0: return tfms

    res_head, res_last = res[:-1], res[-1]
    tfm_head, tfm_rest = tfms[0], tfms[1:]


    subs_map = unify_tuples (res_last, tfm_head)
    return res_head + [mid] + tfm_rest
'''

def norm_tfms_to_shape_pairs (tfms):
    '''
    tfms  'x -> y -> z -> u' or ['x -> y', 'y -> z', 'z -> u'] or ['x -> y -> z', 'z -> u']
    'x', 'y', 'z' are tsns
    returns: [(X, Y), (Y, Z), (Z, U)], where X is the tuple rep for tsn 'x', ...
    '''
    res = []
    if isinstance(tfms, str): 
        shape_pairs = tsnseq2shape_pairs(tfms)
        res.extend(shape_pairs)

    elif isinstance(tfms, (list, tuple)):
        for pos, tfm in enumerate(tfms):
            assert isinstance(tfm, str)
            shape_pairs = tsnseq2shape_pairs(tfm)
            res.extend(shape_pairs)
    else:
        raise ValueError(f'unknown format: {tfms}')

    #print ('norm_tfms', tfms, '\n', res)
    return res


def norm_tfm_names (tfm_names):
    '''
    tfm_names: 'abc' or ['a', 'bc'] 
    returns: ['a', 'b', 'c']
    '''
    res = []
    if isinstance(tfm_names, str): res = list(tfm_names)
    elif isinstance(tfm_names, (list, tuple)):
        for n in tfm_names:
            assert isinstance(n, str)
            res.extend(list(n))
    return res

def tfm_seq_decompose (tfms, tfm_names):
    '''
    Decompose a multi-step transform into basic (view, permute, expand) transforms
    tfms  'btd -> b,t,2,d//2 -> b,2,t,d//2'
    tfm_names 'vp' , i.e., view, then permute transform
    '''
    tfm_symbols = norm_tfm_names(tfm_names) # ['v', 't']
    tfm_symbols_no_c = [n for n in tfm_symbols if n != 'c'] #list without 'c'
    shape_pairs = norm_tfms_to_shape_pairs(tfms)

    #print (len(tfm_symbols_no_c), len(shape_pairs))
    assert len(tfm_symbols_no_c) == (len(shape_pairs)), \
            f"Num of transform steps {len(shape_pairs)} and names {len(tfm_symbols_no_c)} do not match"
    
    tfm_list = [] # (trf symbol, lhs_shape, rhs_shape)

    curr_pos = 0 #count current tfm's position (handle implicit contiguous)
    for sym in tfm_symbols:
        if sym == 'c':   #contiguous transform
            tfm_list.append((sym, None, None))
        else:
            l, r = shape_pairs[curr_pos]
            tfm_list.append((sym, l, r))
            curr_pos += 1

    return tfm_list


def warp (x, tfms, tfm_names, backend=None, debug=False):
    '''
    Perform a multi-step transform on the tensor x
    x: tensor
    tfms:  'btd -> b,t,2,d//2 -> b,2,t,d//2 -> b,2,t,^n,d//2'
    tfm_names: 'vp' [first (v)iew, then (p)ermute transform]
    backend:  either a string('numpy', 'tf', 'torch') or the corresponding backend.<class>
    debug: prints per-step debugging information
    '''

    be = get_backend(backend, x)
    tfm_list = tfm_seq_decompose(tfms, tfm_names)
    #print (f'tfm list {tfm_list}')

    ret = x
    for sym, l, r in tfm_list:
        if debug:
            print(f'*** processing transform.. {sym}\n {l} -> {r}')
        if sym == 'v' or sym == 'r': #view transform
            new_shape = _view_transform(l, r, be.shape(ret))
            ret = be.view(ret, new_shape)
        elif sym == 'p' or sym == 't':
            perm_indices = _permute_transform(l, r)
            ret = be.transpose(ret, perm_indices)
        #elif sym == 'e':
        #    expand_shape = _expand_transform(l, r)
        #    ret = be.expand(ret, expand_shape)
        elif sym == 'a':
            ret = alignto((ret, l), r)
        elif sym == 'c': 
            ret = be.contiguous(ret)
        elif sym == 'j':
            dims = _join_transform(ret, l, r)
            ret = join(ret, dims, backend=be)
        else:
            assert False, f'Invalid transform symbol {sym}'
        if debug:
            print (f'after transform, shape is: {be.shape(ret)}')
    return ret



def tsn_fill_dot_eqn (lhs, placeholders=['_','^','']):

    '''
    construct the full einsum equation for `dot` by adding new unicode symbols
    lhs: ['_d', 'd__']
    returns: lhs2 = ['ad', 'dbc'], rhs = 'abc'
    '''
    assert len(lhs) == 2
    lhs = [tsn_to_str_list(l)[0] for l in lhs] # [['a', 'b', 'c'], ... ]

    #sanity check
    s1, s2 = set(lhs[0]), set(lhs[1])
    common = list(s1.intersection(s2).difference(set(placeholders)))
    assert len(common) == 1

    chars = get_lowercase_symbols(len(s1)+len(s2)-1, common[0]) 

    rhs = []
    cnt = 0
    lhs2 = []
    for l in lhs:
        r = []
        for c in l: 
            if c in placeholders:
                o = chars[cnt]
                rhs.append(o)
                r.append(o)
                cnt += 1
            else: r.append(c)
        r = ''.join(r)
        lhs2.append(r)

    return lhs2, ''.join(rhs), common[0]


def dot (tfm, x, y, backend=None):
    if '->' in tfm: 
        eqn = tfm.replace('.',',')
        #call einsum
    else:
        if '.' not in tfm:
            print ('To avoid confusion, please separate the shorthand shapes by ".", e.g., "_d.d__"')
            if tfm.count(',') == 1:
                tfm = tfm.replace(',','.')
            else:
                raise ValueError(f'Invalid dot transform spec {tfm}')
        
        lp, r = tfm.split('.')
        lp, r, proj = tsn_fill_dot_eqn([lp, r])
        eqn = ','.join(lp) + '->' + r

    #print (f'eqn: {eqn}, {x.size()} {y.size()}')
    be = get_backend(backend, x)
    return be.einsum(eqn, (x, y))

























