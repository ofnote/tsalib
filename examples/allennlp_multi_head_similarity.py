#Original file : https://github.com/allenai/allennlp/blob/master/allennlp/modules/similarity_functions/multiheaded.py

# The annotations in the `forward` function are sufficient to explain the module's functionality

import sys
sys.path.append('../')
from tsalib import dim_vars


from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.common.checks import ConfigurationError
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity


@SimilarityFunction.register("multiheaded")
class MultiHeadedSimilarity(SimilarityFunction):
    """
    This similarity function uses multiple "heads" to compute similarity.  That is, we take the
    input tensors and project them into a number of new tensors, and compute similarities on each
    of the projected tensors individually.  The result here has one more dimension than a typical
    similarity function.
    For example, say we have two input tensors, both of shape ``(batch_size, sequence_length,
    100)``, and that we want 5 similarity heads.  We'll project these tensors with a ``100x100``
    matrix, then split the resultant tensors to have shape ``(batch_size, sequence_length, 5,
    20)``.  Then we call a wrapped similarity function on the result (by default just a dot
    product), giving a tensor of shape ``(batch_size, sequence_length, 5)``.
    Parameters
    ----------
    num_heads : ``int``
        The number of similarity heads to compute.
    tensor_1_dim : ``int``
        The dimension of the first tensor described above.  This is ``tensor.size()[-1]`` - the
        length of the vector `before` the multi-headed projection.  We need this so we can build
        the weight matrix correctly.
    tensor_1_projected_dim : ``int``, optional
        The dimension of the first tensor `after` the multi-headed projection, `before` we split
        into multiple heads.  This number must be divisible evenly by ``num_heads``.  If not given,
        we default to ``tensor_1_dim``.
    tensor_2_dim : ``int``, optional
        The dimension of the second tensor described above.  This is ``tensor.size()[-1]`` - the
        length of the vector `before` the multi-headed projection.  We need this so we can build
        the weight matrix correctly.  If not given, we default to ``tensor_1_dim``.
    tensor_2_projected_dim : ``int``, optional
        The dimension of the second tensor `after` the multi-headed projection, `before` we split
        into multiple heads.  This number must be divisible evenly by ``num_heads``.  If not given,
        we default to ``tensor_2_dim``.
    internal_similarity : ``SimilarityFunction``, optional
        The ``SimilarityFunction`` to call on the projected, multi-headed tensors.  The default is
        to use a dot product.
    """
    def __init__(self,
                 num_heads: int,
                 tensor_1_dim: int,
                 tensor_1_projected_dim: int = None,
                 tensor_2_dim: int = None,
                 tensor_2_projected_dim: int = None,
                 internal_similarity: SimilarityFunction = DotProductSimilarity()) -> None:
        super(MultiHeadedSimilarity, self).__init__()
        self.num_heads = num_heads
        self._internal_similarity = internal_similarity
        tensor_1_projected_dim = tensor_1_projected_dim or tensor_1_dim
        tensor_2_dim = tensor_2_dim or tensor_1_dim
        tensor_2_projected_dim = tensor_2_projected_dim or tensor_2_dim
        if tensor_1_projected_dim % num_heads != 0:
            raise ConfigurationError("Projected dimension not divisible by number of heads: %d, %d"
                                     % (tensor_1_projected_dim, num_heads))
        if tensor_2_projected_dim % num_heads != 0:
            raise ConfigurationError("Projected dimension not divisible by number of heads: %d, %d"
                                     % (tensor_2_projected_dim, num_heads))

        # tsalib dim vars defined locally (to minimize changes from original implementation)
        # better: define and store them in the config dictionary and use everywhere
        self.D1, self.D2, self.D1p, self.D2p = dim_vars('D1:{0} D2:{1} D1p:{2} D2p:{3}'
                        .format(tensor_1_dim, tensor_2_dim, tensor_1_projected_dim, tensor_2_projected_dim))
        
        # original impl
        self._tensor_1_projection = Parameter(torch.Tensor(tensor_1_dim, tensor_1_projected_dim))
        self._tensor_2_projection = Parameter(torch.Tensor(tensor_2_dim, tensor_2_projected_dim))
        
        # with tsalib:
        self._tensor_1_projection: (self.D1, self.D1p) = Parameter(torch.Tensor(self.D1, self.D1p))
        self._tensor_2_projection: (self.D2, self.D2p) = Parameter(torch.Tensor(self.D2, self.D2p))


        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._tensor_1_projection)
        torch.nn.init.xavier_uniform_(self._tensor_2_projection)

    def forward_old(self, tensor_1: 'b,t,d1', tensor_2: 'b,t,d2') :
        # This is the original `forward` implementation
        # note the shape 'surgery' below
        H = self.num_heads
        B, T = dim_vars('Batch(b):{tensor_1.shape(0)} T(t):{tensor_1.shape(1)}')
        D1, D2, D1p, D2p = self.D1, self.D2, self.D1p, self.D2p

        projected_tensor_1: (B, T, D1p) = torch.matmul(tensor_1, self._tensor_1_projection)
        projected_tensor_2: (B, T, D2p) = torch.matmul(tensor_2, self._tensor_2_projection)

        # Here we split the last dimension of the tensors from (..., projected_dim) to
        # (..., num_heads, projected_dim / num_heads), using tensor.view().
        last_dim_size = projected_tensor_1.size(-1) // H
        new_shape = list(projected_tensor_1.size())[:-1] + [H, last_dim_size]
        split_tensor_1: (B, T, H, D1p // H) = projected_tensor_1.view(*new_shape)
        
        last_dim_size = projected_tensor_2.size(-1) // H
        new_shape = list(projected_tensor_2.size())[:-1] + [H, last_dim_size]
        split_tensor_2: (B, T, H, D2p // H) = projected_tensor_2.view(*new_shape)

        # And then we pass this off to our internal similarity function.  Because the similarity
        # functions don't care what dimension their input has, and only look at the last dimension,
        # we don't need to do anything special here.  It will just compute similarity on the
        # projection dimension for each head, returning a tensor of shape (..., num_heads).
        ret : (B, T, H) = self._internal_similarity(split_tensor_1, split_tensor_2)
        return ret

    @overrides
    def forward(self, tensor_1: 'b,t,d1', tensor_2: 'b,t,d2') :
        # Cleaner implementation with tsalib

        #B, T, H defined locally here (to minimize changes to original implementation)
        # better: define and store them in the config dictionary and use everywhere
        B, T, H = dim_vars(f'Batch(b):{tensor_1.shape(0)} T(t):{tensor_1.shape(1)} H(h):{self.num_heads}')
        D1, D2, D1p, D2p = self.D1, self.D2, self.D1p, self.D2p

        projected_tensor_1: (B, T, D1p) = torch.matmul(tensor_1, self._tensor_1_projection)
        projected_tensor_2: (B, T, D2p) = torch.matmul(tensor_2, self._tensor_2_projection)

        split_tensor_1 = projected_tensor_1.view(B, T, H, D1p // H)
        split_tensor_2  = projected_tensor_2.view(B, T, H, D2p // H)

        # And then we pass this off to our internal similarity function.  Because the similarity
        # functions don't care what dimension their input has, and only look at the last dimension,
        # we don't need to do anything special here.  It will just compute similarity on the
        # projection dimension for each head, returning a tensor of shape (..., num_heads).
        ret : (B, T, H) = self._internal_similarity(split_tensor_1, split_tensor_2)
        return ret