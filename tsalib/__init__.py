name = "tsalib"

#import sys
#if sys.version_info < (3, 6, 1):
#    raise RuntimeError("TSAlib requires Python 3.6.1 or later")

from .ts import dim_var, dim_vars, get_dim_vars
from .tsn import tsn_to_shape
from .utils import select, reduce_dims, size_assert, int_shape
from .transforms import view_transform, permute_transform, join_transform
from .transforms import _expand_transform
from .transforms import alignto
from .tensor_ops import warp, join, dot
