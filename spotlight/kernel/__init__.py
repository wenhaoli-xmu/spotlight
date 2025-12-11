from .attn_dx_u8 import attn_dx_u8_func
from .attn_dx_u32 import attn_dx_u32_func

from .pack_dx_u8 import pack_dx_u8_func
from .pack_dx_u32 import pack_dx_u32_func

try:
    from .attn_k4_q28 import attn_k4_q28
    from .packbits import packbits
except ImportError:
    raise ImportError('Should install compile cuda kernel first, please run `bash install.sh`.')


func_mapper = {
    (4, 28): attn_k4_q28,
}


def pack_triton_func(x):
    return pack_dx_u32_func(x)


def pack_cuda_func(x):
    return packbits(x)
    

def attn_triton_func(q_hash, k_hash):
    return attn_dx_u32_func(q_hash, k_hash)


def attn_cuda_func(q_hash, k_hash):
    global func_mapper

    _, _, q_head, head_dim = q_hash.shape
    _, _, k_head, _ = k_hash.shape

    assert head_dim == 4, f'only support head dim = 4 now'
    assert (k_head, q_head) in func_mapper.keys(), f'(k_head, q_head) of {k_head}, {q_head} is not supported'

    return func_mapper[k_head, q_head](q_hash, k_hash)
