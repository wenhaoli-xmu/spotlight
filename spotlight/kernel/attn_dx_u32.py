import triton
import triton.language as tl
import torch


# @triton.jit
# def dx_u8_kernel(
#     q_ptr, k_ptr, o_ptr, 
#     stride_qb,
#     stride_qh,
#     stride_kb,
#     stride_kn,
#     stride_kh,
#     stride_ob,
#     stride_on,
#     N,
#     G: tl.constexpr,
#     KH: tl.constexpr,
#     D: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     # query: (b, 1, 32, d)
#     # key:   (b, 32769, 8, d)

#     b_idx = tl.program_id(0)
#     n_idx = tl.program_id(1)

#     n_range = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
#     h_range = tl.arange(0, KH)
#     d_range = tl.arange(0, D)
    
#     n_mask = n_range < N

#     k_offs = (
#         b_idx * stride_kb + 
#         n_range[:, None, None] * stride_kn + 
#         h_range[None, :, None] * stride_kh + 
#         d_range[None, None, :])
    
#     k_data = tl.load(k_ptr + k_offs, mask=n_mask[:, None, None])

#     accum = tl.zeros((BLOCK_N, KH), dtype=tl.int16)
#     bit_1 = tl.full((1,), value=1, dtype=tl.uint8)

#     for group_idx in range(G):

#         q_offs = (
#             b_idx * stride_qb + 
#             (h_range + group_idx * KH)[:, None] * stride_qh + 
#             d_range[None, :])

#         q_data = tl.load(q_ptr + q_offs)

#         # q_data: (1, 8, d)
#         # k_data: (BLOCK_N, 8, d)

#         # score: (BLOCK_N, 8, d)
#         score = ~(q_data ^ k_data)

#         for _ in range(8):
#             accum += tl.sum(score & bit_1, axis=2).to(tl.int16)
#             score = score >> bit_1

#     o_offs = b_idx * stride_ob + n_range[:, None] * stride_on + h_range[None, :]
#     tl.store(o_ptr + o_offs, accum, mask=n_mask[:, None])


@triton.jit
def dx_u32_kernel(
    q_ptr, k_ptr, o_ptr, 
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_ob,
    stride_on,
    N,
    G: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx = tl.program_id(0)
    n_idx = tl.program_id(1)

    n_range = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    h_range = tl.arange(0, KH)
    d_range = tl.arange(0, D)
    
    n_mask = n_range < N

    k_offs = (
        b_idx * stride_kb + 
        n_range[:, None, None] * stride_kn + 
        h_range[None, :, None] * stride_kh + 
        d_range[None, None, :])
    
    k_data = tl.load(k_ptr + k_offs, mask=n_mask[:, None, None])
    accum = tl.zeros((BLOCK_N, KH), dtype=tl.int16)

    # Precompute masks for popcnt calculation
    mask_0x55555555u = tl.full((1,), 0x55555555, dtype=tl.uint32)
    mask_0x33333333u = tl.full((1,), 0x33333333, dtype=tl.uint32)
    mask_0x0f0f0f0fu = tl.full((1,), 0x0f0f0f0f, dtype=tl.uint32)
    mask_0x00ff00ffu = tl.full((1,), 0x00ff00ff, dtype=tl.uint32)
    mask_0x0000ffffu = tl.full((1,), 0x0000ffff, dtype=tl.uint32)

    for group_idx in tl.static_range(G):
        q_offs = (
            b_idx * stride_qb + 
            (h_range + group_idx * KH)[:, None] * stride_qh + 
            d_range[None, :])
        q_data = tl.load(q_ptr + q_offs)

        score = ~(q_data ^ k_data)

        score = (score & mask_0x55555555u) + ((score >> 1) & mask_0x55555555u)
        score = (score & mask_0x33333333u) + ((score >> 2) & mask_0x33333333u) 
        score = (score & mask_0x0f0f0f0fu) + ((score >> 4) & mask_0x0f0f0f0fu)
        score = (score & mask_0x00ff00ffu) + ((score >> 8) & mask_0x00ff00ffu)
        score = (score & mask_0x0000ffffu) + ((score >>16) & mask_0x0000ffffu)

        score = tl.sum(score, axis=2)
        accum += score.to(tl.int16)

    o_offs = b_idx * stride_ob + n_range[:, None] * stride_on + h_range[None, :]
    tl.store(o_ptr + o_offs, accum, mask=n_mask[:, None])


def attn_dx_u32_func(q_hash, k_hash):
    BLOCK_N = 64

    B, QN, QH, D = q_hash.shape
    _, KN, KH, _ = k_hash.shape

    assert QN == 1, f"only support single query"
    assert QH % KH == 0, f"query head must be multiple of key head"

    output = torch.zeros(
        (B, KN, KH), 
        dtype=torch.int16, 
        device=q_hash.device)
    
    assert q_hash.is_contiguous()
    assert k_hash.is_contiguous()

    grid = (B, triton.cdiv(KN, BLOCK_N))

    with torch.cuda.device(q_hash.device):
        dx_u32_kernel[grid](
            q_hash, k_hash, output,
            q_hash.stride(0), q_hash.stride(2),
            k_hash.stride(0), k_hash.stride(1), k_hash.stride(2),
            output.stride(0), output.stride(1),
            N=KN,
            G=QH//KH,
            KH=KH,
            D=D,
            BLOCK_N=BLOCK_N)
    
    return output