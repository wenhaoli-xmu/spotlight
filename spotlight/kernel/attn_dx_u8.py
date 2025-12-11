import triton
import triton.language as tl
import torch


table = torch.tensor([bin(i).count('1') for i in range(256)], dtype=torch.uint8)


@triton.jit
def dx_u8_kernel(
    q_ptr, k_ptr, o_ptr,
    stride_qb,
    stride_kb,
    stride_kn,
    stride_ob,
    B, N, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx, n_idx = tl.program_id(0), tl.program_id(1)

    n_range = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, D)

    bit_1 = tl.full((1,), value=1, dtype=tl.uint8)

    # load query
    q_offs = b_idx * stride_qb + d_range[None, :]
    q_data = tl.load(q_ptr + q_offs)

    # load key
    k_offs = b_idx * stride_kb + n_range[:, None] * stride_kn + d_range[None, :]
    k_data = tl.load(k_ptr + k_offs, mask=k_offs < B * N * D)

    # lsh attention
    xor_result = ~(q_data ^ k_data)
    accum = tl.zeros((BLOCK_N,), dtype=tl.uint8)

    for _ in range(8):
        accum += tl.sum(xor_result & bit_1, axis=1)
        xor_result = xor_result >> bit_1

    o_offs = b_idx * stride_ob + n_range
    tl.store(o_ptr + o_offs, accum, mask=o_offs < B * N)


def attn_dx_u8_func(q_hash, k_hash):
    BLOCK_N = 128
    batch_size, num_heads, q_len, dim = q_hash.shape
    _, _, k_len, _ = k_hash.shape
    
    assert q_len == 1, f"only support single query"
    
    q_hash = q_hash.view(-1, dim)  
    k_hash = k_hash.view(-1, k_len, dim) 

    result = torch.zeros((q_hash.shape[0], k_len), dtype=torch.uint8, device=q_hash.device)
    
    q_hash = q_hash.contiguous()
    k_hash = k_hash.contiguous()
    grid = (batch_size * num_heads, triton.cdiv(k_len, BLOCK_N))

    with torch.cuda.device(q_hash.device):
        dx_u8_kernel[grid](
            q_hash, k_hash, result,
            q_hash.stride(0),
            k_hash.stride(0),
            k_hash.stride(1),
            result.stride(0),
            k_hash.shape[0], 
            k_hash.shape[1], 
            k_hash.shape[2],
            BLOCK_N)
    
    return result.view(batch_size, num_heads, k_len)


# def attn_dx_u8_func(q_hash, k_hash):
#     global table
#     table = table.to(q_hash.device, non_blocking=True)

#     q_hash = q_hash.unflatten(1, (k_hash.shape[1], -1))
#     k_hash = k_hash[:, :, None, :, :]

#     sim = ~torch.bitwise_xor(q_hash, k_hash).flatten(1,2)
#     sim_shape = sim.shape
#     sim = table[sim.view(-1).int()].view(*sim_shape).sum(-1)

#     return sim