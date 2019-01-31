from .tsn import tsn_to_str_list, tsn_to_tuple

def get(x, dv_dict):
    '''
    Index using dimension shorthands
    
    x: arbitrary tensor (can be indexed in numpy notation : x[:,0,:])
    dv_dict: {'b': 0, 'c': 5} 
    x_tsa: 'b,c,d'
    '''

    assert isinstance(tuple), 'The first argument should be a tuple of (vector, shape)'
    xv, xs = x
    shape, is_seq = tsn_to_str_list(xs)
    if not is_seq:
        raise NotImplementedError(f"get from shape {xs} not implemented")

    colon = slice(None)
    slice_tuple = []
    for sh in shape:
        if sh in dv_dict:
            slice_tuple.append(dv_dict[sh])
        else:
            slice_tuple.append(colon)

    y = x[slice_tuple]
    return y

def size_assert(x_size, sa, dims=None):
    '''
    x_size: integer tuple
    sa: TSA
    dims: None or Sequence[int], e.g., [0,1]
    Check if size of tensor x matches TSA `sa` along `dims` axes
    '''
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

