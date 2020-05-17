
# coding: utf-8

# In[2]:


import sys
import numpy as np
from tsalib import dim_vars, get_dim_vars, update_dim_vars_len


# # Design Principles
# **Dimension Variables** (DVs) are the core abstractions behind tsalib. 
# - They allow specifying and modifying shapes of tensors *symbolically*, i.e., using named symbols corresponding to different dimensions of tensor. 
# - Making dimension names explicit enables cleaner, DRY code, symbolic shape assertions, and faster debugging.
# - **Symbolic** shapes or **annotations** are *tuples* over DVs and arithmetic expressions over DVs.
# 
# The `tsalib` provides a collection of powerful APIs to handle all kinds of shape transformations using explicit dimension variables and shape annotations.  
# 
# 
# - Designed to stay light, easy to incorporate into existing workflow with minimal code changes.
# - The API includes both library-independent and dependent parts, giving developers flexibility in how they choose to incorporate `tsalib` in their workflow.
# - Avoid deeper integration into popular tensor libraries to keep `tsalib` light-weight and avoid backend-inflicted bugs.
# 
# Some popular models (resnet, transformer) annotated/re-written with tsalib can be found in the [models](models/) directory.
# 

# ## Declare dimension variables
# Dimension variables model both the `name` and the default `size` of a tensor.   
# Format: **name(symbol):size**   --  `symbol` and `size` are optional
# 
# We can declare dimension variables **globally** (Dimensions used in programs are known upfront and programs don't modify dimension names).  
# Even better, we can put all these definitions in the Config dictionary.

# In[3]:


# globals variables prefixed with underscores
_B, _T, _D, _K = dim_vars('Batch(b):20 SeqLength(t):10 EmbeddingDim(d):100 K(k):1')
_C, _H, _W = dim_vars('Channels(c):3 Height(h):256 Width(w):256')


# In[4]:


def test_decls():
    print('\nTest declarations ..')
    #local declarations
    print(f'B, C, D = {_B}, {_C}, {_D}')

    #strict=False allows overwriting previous declarations
    H, W = dim_vars ('Height(h):256 Width(w):256', exists_ok=True) 
    print(f'H, W = {H}, {W}')

    # test update dim var len

    H.update_len(1024)
    print(f'H = {H}')

    update_dim_vars_len({'h': 512, 'w': 128})
    H, W = get_dim_vars('h w')
    print(f'H, W = {H}, {W}')




# Supports arithmetic over a combination of dim vars and other Python variables
def test_arith():
    print('\nTest arithmetic ..')
    _K, _W, _B, _H = get_dim_vars('k w b h') 
    _K = _W * 2
    h = 4
    print((h, _H // h, _K, _B*2))

# Use dimension variables in lieu of constant size values
# note: dim_var declaration must include size of the variable
def test_cast_int():
    print('\nTest integer cast ..')
    B, C = get_dim_vars('b c')
    x = np.zeros((B, C))
    print(f'shape of array: ({B},{C}): {x.shape}')
    return x
    
def basic_tests():
    test_decls()
    test_arith()
    x = test_cast_int()
    # Test assertions over symbolic shapes
    assert x.shape == (_B,_C)
    print ('assertions hold')


# In[5]:


basic_tests()


# ## Basic tsalib usage
# Can be used to manage tensor shapes with **arbitrary** tensor libraries. Here, examples with *numpy* and *pytorch*.
# - Create new tensors (independent of actual dimension sizes)
# - **Annotate** tensor variables (widely considered best practice, otherwise done using adhoc comments)
# - Check symbolic **assertions** (assertions **do not** change even if dimension size changes)

# In[6]:


def test_numpy():
    print('\nTest usage with numpy ..')
    B, D = get_dim_vars('b d')
    import numpy as np
    a: (B, D) = np.zeros((B,D))
    print(f'original array: {(B,D)}: {a.shape}')

    b: (2, B, D) = np.stack([a, a])
    print(f'after stack: {(2,B,D)}: {b.shape}')

    ax = (2,B,D).index(B)
    c: (2, D) = np.mean(b, axis=ax)
    print(f'after mean along axis = {ax}: {(2,D)}: {c.shape}')

test_numpy()


# In[7]:


def test_pytorch():
    print('\nTest usage with pytorch ..')
    B, D = get_dim_vars('b d')
    B, D = dim_vars('Batch:2 EmbedDim:3', exists_ok=True)
    import torch

    a = torch.Tensor([[1., 2., 4.], [3., 6., 9.]])
    assert a.size() == (B, D)

    b = torch.stack([a, a])

    print ('Asserting b.size() == (2,B,D)')
    assert b.size() == (2, B, D)

    c = torch.cat([a, a], dim=1)
    print ('Assertion on c.size()')
    assert c.size() == (B, D*2)

test_pytorch()


# ## Shape Transformations with Dimensions Variables
# To shape transform without `tsalib`, you either 
# -  **hard-code** integer constants for each dimension's position in shape transformations, or
# - do shape tuple **surgeries** to compute the 'right' shape (for the general case)
# 
# Instead, with `tsalib`, use dimension variables or the shorthand symbols directly. 
# 
# `tsalib` provides API for common shape transformations: **view** (reshape), **permute** (transpose) and **expand** (tile).  
# These are *library-independent*, e.g., shorthand transformation -> target shape tuple -> reshape.
# 
# One transformation to rule them all : **warp**. Do a sequence of transformations on a tensor.  
# `warp` is implementated for several popular backend libraries.
# 
# ## Work with Shorthand Shape Notation 
# Writing tuples of shape annotations can get cumbersome.
# 
# So, instead of (B, T, D), write 'btd' (each dim gets a single char, concatenated together)
# 
# Instead of (B \* T, D // 2, T), write 'b * t, d//2, t' (arbitrary arithmetic expressions, comma-separated)
# 
# Anonymous dimension variables : 'b,,d' omits naming dimension t.

# ## Reshapes (view transformations) using dimension variables
# These are library independent: `vt` returns target tensor shapes from shorthand transformation spec.

# In[8]:


# without tsalib, this is how we used to do it. See code from BERT.
def test_reshape_old ():
    x = np.ones((20, 10, 100))
    h = 4
    new_shape = x.shape[:2] + (h, x.shape[2]//h) #shape surgery
    x = x.reshape(new_shape)
    print (x.shape)

from tsalib import view_transform as vt    
    
# with tsalib, simply use dimension vars in-place
def test_reshape():
    B, T, D = get_dim_vars('b t d')
    x: (B,T,D) = np.ones((B, T, D))
    h = 4
    x: (B,T,h,D//h) = x.reshape((B, T, h, D//h))
    assert x.shape == (B,T,h,D//h)
    print ('test_reshape: all assertions hold')

#using shorthand notation, omit dimensions not involved in transformation
def test_reshape_short():
    B, T, D = get_dim_vars('b t d')
    x: (B,T,D) = np.ones((B, T, D))
    h = 4
    x = x.reshape(vt(f'btd -> b,t,{h},d//{h}', x.shape))
    assert x.shape == (B, T, h, D//h)

    x1 = x.reshape(vt('b,t,4,k -> b*t,4,k', x.shape))
    assert x1.shape == (B*T, h, D//h)
    
    x1 = x.reshape(vt('b,t,, -> b*t,,', x.shape))
    assert x1.shape == (B*T, h, D//h)


    print ('test_reshape_short: all assertions hold')


#test_reshape_old()
test_reshape()
test_reshape_short()


# ## Transpose/Permute transformations using dimension variables

# In[9]:


from tsalib import  permute_transform as pt
from tsalib.transforms import _permute_transform as _pt

# permute using dimension variables (internal, recommended to be not used)
def test_permute():
    B, T, D, K = get_dim_vars('b t d k')
    x: (B,T,D,K) = np.ones((B, T, D, K))
    perm_indices = _pt(src=(B,T,D,K), to=(D,T,B,K))
    assert perm_indices == (2,1,0,3)
    x = x.transpose(perm_indices)
    assert x.shape == (D,T,B,K)
    print ('test_permute: all assertions hold')

# shorthand permutes are snazzier (use '_' or ',' as placeholders)
def test_permute_short():
    B, T, D, K, C, H, W = get_dim_vars('b t d k c h w')
    x: (B,T,D,K) = np.ones((B, T, D, K))  
    x = x.transpose(pt('btdk -> dtbk')) # (B, T, D, K) -> (D, T, B, K)
    assert x.shape == (D,T,B,K)

    x = x.transpose(pt('d_b_ -> b_d_')) # (D,T,B,K) -> (B, T, D, K)
    assert x.shape == (B,T,D,K)

    x: (B, C, H, W) = np.ones((B, C, H, W))
    x1 = x.transpose(pt(',c,, -> ,,,c'))
    assert x1.shape == (B, H, W, C)
    print ('test_permute_short: all assertions hold')
test_permute()
test_permute_short()


# ## Expand transformations

# In[10]:


from tsalib import _expand_transform as et
def test_expand():
    B, T, D, K = get_dim_vars('b t d k')
    
    x: (B, T, D) = np.ones((B, T, D))
    x: (B, K, T, D) = x[:, None]

    expand_shape = et(src=(B,K,T,D), expansions=[(K, K*5)], in_shape=x.shape) #(B, K, T, D) -> (B, K*5, T, D)
    assert expand_shape == (-1,5,-1,-1)
    print ('test_expand: all assertions hold')

def test_expand_short():
    B, T, D, K = get_dim_vars('b t d k')
    
    x: 'btd' = np.ones((B, T, D))
    x: 'bktd' = x[:, None]
    expand_shape = et(src=(B,K,T,D), expansions='k->k*5', in_shape=x.shape)
    assert expand_shape == (-1,5,-1,-1)
    print ('test_expand_short: all assertions hold')
test_expand()
test_expand_short()


# ## *warp* : generalized shape transformations
# 
# Writing a sequence of shape transformations in code can get cumbersome.  
# `warp` enables specifying a sequence of transformations together **inline**.

# In[11]:


from tsalib import warp
def test_warp():
    B, T, D = get_dim_vars('b t d')
    
    x: 'btd' = np.ones((B, T, D))
    
    # two view transformations (reshapes) in sequence
    x1 = warp(x, 'btd -> b,t,4,d//4 -> b*t,4,d//4', 'vv', debug=False)
    assert(x1.shape == (B*T,4,D//4))

    # four reshapes in sequence
    x2 = warp(x, 'btd -> b,t,4,d//4 -> b*t,4,d//4 -> b,t,4,d//4 -> btd', 'vvvv', debug=False)
    assert(x2.shape == (B,T,D))
    
    # Same reshape sequence in shorthand, specified as list of transformations
    x2 = warp(x, ['__d -> ,,4,d//4', 'b,t,, -> b*t,,', 'b*t,, -> b,t,,', ',,4,d//4 -> ,,d'], 'vvvv', debug=True)
    assert(x2.shape == (B,T,D))
    
    print ('test_warp: all assertions hold')
    

def test_warp_pytorch():
    B, T, D = get_dim_vars('b t d')
    
    import torch
    y: 'btd' = torch.randn(B, T, D)
    #a reshape followed by permute
    y = warp(y, 'btd -> b,t,4,d//4 -> b,4,t,d//4', 'vp', debug=False)
    assert(y.shape == (B,4,T,D//4))

    print ('test_warp_pytorch: all assertions hold')
    
test_warp()
test_warp_pytorch()


# ## Join: unified stack/concatenate for a list of tensors
# Crisp shorthand : `'(b,t,d)* -> b,3*t,d'` (**concat**) or `'(b,t,d)* -> b,^,t,d'` (**stack**)

# In[12]:


from tsalib import join, join_transform
def test_join ():
    B, T, D = get_dim_vars('b t d')
    x1: 'btd' = np.ones((B, T, D))
    x2: 'btd' = np.ones((B, T, D))
    x3: 'btd' = np.ones((B, T, D))
    
    #concatenate along the (T) dimension: (b,t,d)* -> (b,3*t,d)
    x = join([x1, x2, x3], dims=',*,') 
    assert x.shape == (B, 3*T, D)

    
    #stack: join by adding a new dimension to the front: (b,t,d)* -> (^,b,t,d)
    x = join([x1, x2, x3], dims='^') 
    assert x.shape == (3, B, T, D)
    
    #stack by adding a new dimension at second position: (b,t,d)* -> b,^,t,d)
    x = join([x1, x2, x3], dims=',^') 
    assert x.shape == (B, 3, T, D)
    print ('test_join: all assertions passed')
    
    
def test_join_transform():
    B, T, D = get_dim_vars('b t d')
    x1: 'btd' = np.ones((B, T, D))
    x2: 'btd' = np.ones((B, T, D))
    x3: 'btd' = np.ones((B, T, D))
    
    dims = join_transform([x1,x2,x3], '(b,t,d)* -> b,3*t,d')
    assert dims == ',*,'
    #now use backend-dependent join
    
    dims = join_transform([x1,x2,x3], '(b,t,d)* -> b,^,t,d')
    assert dims == ',^,,'
    #now use backend-dependent join
    
    print ('test_join_transform: all assertions passed')
    
test_join()
test_join_transform()


# ## Align one tensor to another

# In[13]:


from tsalib import alignto
def test_align():
    B, T, D = dim_vars('Batch(b):20 SeqLength(t):10 EmbeddingDim(d):100', exists_ok=True)
    
    x1 = np.random.randn(D,D)
    x2 = np.random.randn(B,D,T,D)

    x1_aligned = alignto( (x1, 'dd'), 'bdtd')
    assert x1_aligned.shape == (1,D,1,D)
    print ('test align: all assertion passed')
test_align()


# ## Dot Product of two tensors (sharing exactly one dimension)

# In[14]:


from tsalib import dot
import torch
def test_dot():
    B, C, T, D = get_dim_vars('b c t d')
    #x = np.random.rand(B, C, T)
    #y = np.random.rand(C, D)
    x = torch.randn(B, C, T)
    y = torch.randn(C, D)
    z = dot('_c_.c_', x, y)
    assert z.shape == (B, T, D)
    print('test_dot: all assertions passed')
test_dot()


# # Reduce ops (min, max, mean, ..) with tsalib
# Reduction operators aggregate values over one or more tensor dimensions.  
# `tsalib` provides `reduce_dims` to compute dimension ids using shorthand notation.

# In[15]:


from tsalib import reduce_dims as rd

def test_reduce ():
    assert rd('2bd->2d') == (1,)
    assert rd('2bd->2') == (1,2)
    print ('test_reduce: all assertions hold')
test_reduce()


# In[16]:


x: 'btd' = np.random.rand(_B, _T, _D)
y = np.mean(x, axis=rd('btd->b'))
assert y.shape == (_B,)


# ## Looong warps

# In[17]:


def warp_long1 ():
    B, T, D, C = get_dim_vars('b t d c')
    x1: 'btd' = np.ones((B, T, D))
    x2: 'btd' = np.ones((B, T, D))
    x3: 'btd' = np.ones((B, T, D))
    y = warp([x1,x2,x3], '(btd)* -> btdc -> bdtc -> b,d//2,t*2,c', 'jpv')
    assert y.shape == (B, D//2, T*2, C)
    print ('warp_long1: all assertions hold')
    
def warp_long2 ():
    B, T, D, C = get_dim_vars('b t d c')
    x1: 'btd' = np.ones((B, T, D))
    y = warp(x1, 'btd -> btd1 -> bdt1 -> b,d//2,t*2,1', 'apv')
    assert y.shape == (B, D//2, T*2, 1)
    print ('warp_long2: all assertions hold')
    
    
warp_long1()
#warp_long2()

