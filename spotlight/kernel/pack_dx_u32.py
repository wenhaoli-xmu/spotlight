import triton 
import triton.language as tl
import torch


@triton.jit
def pack_dx_u32_kernel(
    x_ptr, o_ptr,
    stride_xn,
    stride_xd,
    stride_on,
    N, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    last_range = tl.arange(0, 32)
    d_range = tl.arange(0, D)
    n_range = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

    shift = tl.arange(0, 32).to(tl.uint32)
    shift = tl.view(shift, (1, 1, 32))

    n_mask = n_range < N

    # manipulate x
    x_offs = n_range[:, None, None] * stride_xn + d_range[None, :, None] * stride_xd + last_range[None, None, :]
    x_data = tl.load(x_ptr + x_offs, mask=n_mask[:, None, None])
    x_data = (x_data > 0).to(tl.uint32) << shift
    x_data = tl.sum(x_data, axis=2)

    # write back to HBM
    o_offs = n_range[:, None] * stride_on + d_range[None, :]
    tl.store(o_ptr + o_offs, value=x_data.to(tl.uint32), mask=n_mask[:, None])


def pack_dx_u32_func(x):
    last_dim = x.shape[-1]
    other_dims = x.shape[:-1]

    assert last_dim % 32 == 0, f"suppose `last_dim % 8 == 0`"

    BLOCK_N = 64

    assert x.is_contiguous()
    x = x.view(-1, last_dim // 32, 32)
    o = torch.zeros(x.shape[:-1], dtype=torch.uint32, device=x.device)

    grid = (triton.cdiv(x.shape[0], BLOCK_N),)

    with torch.cuda.device(x.device):
        pack_dx_u32_kernel[grid](
            x, o,
            x.stride(0),
            x.stride(1),
            o.stride(0),
            x.shape[0], 
            x.shape[1],
            BLOCK_N)

    return o.view(*other_dims, last_dim // 32).contiguous()


if __name__ == '__main__':
    x = torch.randn((4096, 128), device='cuda') > 0
    result = pack_dx_u32_func(x)
    print(result)