from tsalib.ts import TS


def view_transform (src, to, in_shape):
    '''
    View Transform
    :src is the source view, each element is a single named dim variable
    :to is the target view, may contain expressions over named dim variables
    :in_shape is the shape of the source tensor
    :returns the new size of the tensor after view transformation
    '''
    assert isinstance(src, (list, tuple))
    assert isinstance(to, (list, tuple))
    assert isinstance(in_shape, (list, tuple))

    #sub_map = [(d.e, Symbol(f'{i}')) for i, d in enumerate(src)]
    sub_map = [(d.exp, in_shape[i]) for i, d in enumerate(src)]
    out_shape = tuple([t.exp.subs(sub_map) if isinstance(t, TS) else t for t in to])

    return out_shape




def permute_transform(src, to):
    '''
    Permute Transform
    :src is the source dimension arrangement, each element is a single named dim variable
    :to is the target dimension arragement, each element is a single named dim variable
    :returns the index tuple for the permutation
    '''
    assert isinstance(src, (list, tuple))
    assert isinstance(to, (list, tuple))
    sub_map = [(d.exp, i) for i, d in enumerate(src)]
    perm_indices = tuple([t.exp.subs(sub_map) for t in to])
    return perm_indices



def expand_transform (src, expansions, in_shape):
    '''
    :src        (B, T, D) = (10, 20, 300)
    :expansions [(T, T*5), (D, D*4)]
    :returns the expansion shape tuple
    '''
    assert isinstance(src, (list, tuple))
    assert isinstance(expansions, (list, tuple))

    exp_map = dict(expansions)
    sub_map = [(d.exp, in_shape[i]) for i, d in enumerate(src)] # (B, 10), (T, 20), (D, 300)


    res = []
    for k in src:
        if k not in exp_map: res.append(-1) #keep dim shape same
        else:
            v = exp_map[k]
            assert isinstance(v, TS)
            res.append(v.exp.subs(sub_map)) #(T*5) -> 100, (D*4) -> 1200

    return tuple(res)






