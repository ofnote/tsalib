from .tsn import tsn_to_tuple, tsn_to_str_list
from .backend import get_backend_by_name, get_backend_for_tensor
from .transforms import _view_transform, _permute_transform, _join_transform, _expand_transform
from .utils import get_lowercase_symbols

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


def tfm_seq_decompose (tfms, tfm_names):
    '''
    Decompose a multi-step transform into basic (view, permute, expand) transforms
    tfms  'btd -> b,t,2,d//2 -> b,2,t,d//2'
    tfm_names 'vp' , i.e., view, then permute transform
    '''
    tfm_symbols = list(tfm_names) # ['v', 't']
    tfm_symbols_no_c = list(tfm_names.replace('c',''))
    tfm_list = [] # (trf symbol, trf_lhs, trf_rhs)

    if isinstance(tfms, str):

        shapes = [t.strip() for t in tfms.split('->')]
        assert len(shapes) >= 2, "Specify at least one transform, e.g., btd->dtb"
        assert len(tfm_symbols_no_c) == (len(shapes) - 1), "Num of transform descriptions and symbols do not match"

        shapes = [tsn_to_tuple(s) for s in shapes]
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
            ret = be.view(x, new_shape)
        elif sym == 'p' or sym == 't':
            perm_indices = _permute_transform(l, r)
            ret = be.transpose(ret, perm_indices)
        #elif sym == 'e':
        #    expand_shape = _expand_transform(l, r)
        #    ret = be.expand(ret, expand_shape)
        elif sym == 'c': 
            ret = be.contiguous(ret)
        elif sym == 'j':
            dims = _join_transform(ret, l, r)
            ret = join(ret, dims, backend=be)
        else:
            assert False, f'Invalid transform symbol {sym}'
        if debug:
            print (f'   after transform shape is: {be.shape(ret)}')
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
    if '->' in tfm: #call einsum 
        #raise NotImplementedError('todo: call einsum directly')
        eqn = tfm.replace('.',',')
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

    #print (f'eqn: {eqn}')
    be = get_backend(backend, x)
    return be.einsum(eqn, (x, y))

























