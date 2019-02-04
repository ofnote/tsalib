import torch
import torch.nn.functional as F


import sys
sys.path.append('../')
from tsalib import dim_vars as dvs, get_dim_vars
from tsalib import permute_transform as pt, warp, dot, alignto
from tsalib import reduce_dims as rd

B, H, T, D = dvs('Batch(b):4 H(h):7 T(t):100 D(d):300')


# `merge_heads` function in Transformer network (original)

def merge_heads_old(x: (B,H,T,D)):
  x = x.permute(0, 2, 1, 3).contiguous()
  new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
  res = x.view(*new_x_shape)
  return res


# `merge_heads` using tsalib (weaker integration)


def merge_heads1(x: (B,H,T,D)):
  x: (B,T,H,D) = x.permute(pt('bhtd -> bthd')).contiguous()
  res: (B,T,H*D) = x.view((B,T,H*D))
  return res


# 'merge_heads' using tsalib's warp (deeper integration)

from tsalib import warp

def merge_heads2(x: (B,H,T,D)):
    res: (B,T,H*D) = warp(x, 'bhtd -> bthd -> b,t,h*d', 'pv', debug=True)
    return res


def test_merge_heads():
    x = torch.randn( (B,H,T,D) )
    y = merge_heads_old(x)
    assert y.size() == (B,T,H*D)
    y = merge_heads1(x)
    assert y.size() == (B,T,H*D)
    
    y = merge_heads2(x)
    assert y.size() == (B,T,H*D)
    print ('all merge_heads assertions hold')


'''
Einsum attention 

Originally from https://rockt.github.io/2018/04/30/einsum

Revisited at http://nlp.seas.harvard.edu/NamedTensor

'''


def random_tensors(shape, num=1):
  tensors = [torch.randn(shape) for i in range(0, num)]
  return tensors[0] if num == 1 else tensors

def make_params (l):
    bM, br, w = random_tensors([l], num=3)
    # -- [hidden_dimension x hidden_dimension]
    WY, Wh, Wr, Wt = random_tensors([l, l], num=4)

    return (bM, br, w), (WY, Wh, Wr, Wt)

def einsum_attn(Y, ht, rt1):
    (bM, br, w), (WY, Wh, Wr, Wt) = make_params(7)

    # -- [batch_size x hidden_dimension]
    tmp = torch.einsum("ik,kl->il", [ht, Wh]) + \
          torch.einsum("ik,kl->il", [rt1, Wr])

    Mt = torch.tanh(torch.einsum("ijk,kl->ijl", [Y, WY]) + \
                tmp.unsqueeze(1).expand_as(Y) + bM)
    # -- [batch_size x sequence_length]
    at = F.softmax(torch.einsum("ijk,k->ij", [Mt, w]), dim=-1)

    # -- [batch_size x hidden_dimension]
    rt = torch.einsum("ijk,ij->ik", [Y, at]) + \
         torch.tanh(torch.einsum("ij,jk->ik", [rt1, Wt]) + 
                    br)

    # -- [batch_size x hidden_dimension], [batch_size x sequence_dimension]
    return rt, at

def test_einsum_attn():
    # -- [batch_size x sequence_length x hidden_dimension]
    Y = random_tensors([3, 5, 7])
    # -- [batch_size x hidden_dimension]
    ht, rt1 = random_tensors([3, 7], num=2)

    rt, at = einsum_attn(Y, ht, rt1)
    assert rt.size() == (3, 7) and at.size() == (3, 5)

    print ('einsum attn: assertions hold')


'''
With tsalib: 
'''

def tsa_attn(Y, ht, rt1):
    B, L, D = get_dim_vars('b l d')
    Y: 'bld' ; ht: 'b,d'; rt1: 'b,d'

    #bM, br, w: 'd,'
    #WY, Wh, Wr, Wt: 'd,d' 
    (bM, br, w), (WY, Wh, Wr, Wt) = make_params(D)

    tmp: 'bd' = dot(ht, Wh, '_d.d_') + dot(rt1, Wr, '_d.d_')
    tmp: 'bld' = alignto((tmp,'bd'), (Y,'bld'), expand=True)

    Mt: 'bld' = torch.tanh(dot(Y, WY, '__d.d_') + tmpa + bM)
    at: 'bl' = F.softmax(dot(Mt, w, '__d.d'), dim=-1)
    rt: 'bd' = dot(Y, at, 'bld,bl->bd') + torch.tanh(dot(rt1, Wt, '_d.d_') + br)

    return rt, at

def test_tsa_attn():
    B, L, D = dvs('Batch(b):3, sequence_length(l):5 hidden_dimension(d):7', check=False)

    Y = random_tensors([B, L, D])
    ht, rt1 = random_tensors([B, D], num=2)
    rt, at = tsa_attn(Y, ht, rt1)

    assert rt.size() == (B, D) and at.size() == (B, L)
    print ('tsa attn: assertions hold')

if __name__ == '__main__':
    #test_merge_heads()
    #test_einsum_attn()
    test_tsa_attn()
