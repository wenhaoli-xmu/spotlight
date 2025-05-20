import math
import torch
import torch.utils.checkpoint
from transformers.models.llama.modeling_llama import rotate_half
from functools import partial
from flash_attn import flash_attn_func


def segment(tensor, dim, n):
    total_length = tensor.shape[dim]

    for start in range(0, total_length, n):
        end = min(start + n, total_length)
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(start, end)
        yield tensor[tuple(indices)]


def compute_loss(logits, labels, shift=False):
    """
    Returns:
        token_loss: batch_size, seq_length
    """
    if shift:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    labels = labels.to(logits.device)
    batch_size = logits.shape[0]

    # NOTE: the loss on -100 labels is 0 by default
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        labels.reshape(-1), 
        reduction="none"
    ).reshape(batch_size, -1)   # batch_size, seq_len
    
    valid_token_num = (labels != -100).sum(-1)  # batch_size
    all_valid_token_num = valid_token_num.sum()
    
    if all_valid_token_num > 0:
        loss = token_loss.sum() / valid_token_num.sum()
    else:
        loss = token_loss.sum()

    batch_loss = token_loss.sum(-1) / valid_token_num
    # prevent nan
    if (valid_token_num == 0).any():
        batch_loss = batch_loss.masked_fill(valid_token_num == 0, 0.)

    return loss, batch_loss, valid_token_num


def apply_rotary_pos_emb(mat, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    mat_embed = (mat * cos) + (rotate_half(mat) * sin)

    return mat_embed


def new_posid(num_token: int, device, dtype, bsz):
    appendix = torch.arange(num_token, device=device)
    appendix = appendix[None,:].expand(bsz, -1)
    return appendix


def check_and_apply_qk_rope(query, key, cos, sin, pos=0):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)
    pos_list = new_posid_spec(max(num_kv, pos))

    Q = apply_rotary_pos_emb(query, cos, sin, pos_list[:,-num_query:])
    K = apply_rotary_pos_emb(key, cos, sin, pos_list[:,-num_kv:])

    return Q, K


def check_and_apply_qk_rope_random_query(query, key, cos, sin, query_index: torch.LongTensor):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert query_index.ndim == 1, f"{query_index.shape}"

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    Q = apply_rotary_pos_emb(query, cos, sin, new_posid_spec(num_kv)[:,query_index])
    K = apply_rotary_pos_emb(key, cos, sin, new_posid_spec(num_kv))

    return Q, K


def check_and_apply_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    Q = apply_rotary_pos_emb(query, cos, sin, new_posid_spec(num_kv)[:,-num_query:])
    K = apply_rotary_pos_emb(key, cos, sin, new_posid_spec(num_kv))
    V = value

    return Q, K, V


def generate_decoder_mask(num_querys, num_keys, dtype, device, debug=False):
    assert num_querys <= num_keys
    mask = torch.full((1,1,num_querys,num_querys), torch.finfo(dtype).min, device=device, dtype=torch.float32).triu(diagonal=1).type(dtype)
    prefix = torch.zeros((1,1,num_querys,num_keys-num_querys), device=device, dtype=dtype)
    mask = torch.cat([prefix, mask], dim=-1)

    assert mask.shape == (1, 1, num_querys, num_keys)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].cpu())
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='In generate_decoder_mask')

    assert (mask != 0).sum().item() == num_querys * (num_querys - 1) / 2
    assert (mask == 0).sum().item() == num_querys * num_keys - num_querys * (num_querys - 1) / 2

    return mask


def generate_mask(num_query, num_kv, dtype, device):
    mask = torch.full(
        (1, 1, num_query, num_kv), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )
    assert num_query <= num_kv
    mask[0,0,:,-num_query:].triu_(diagonal=1)
    mask[0,0,:,:-num_query].fill_(0)
    mask = mask.type(dtype)
    return mask


def get_attn_score(query, key, cos, sin):
    Q, K = check_and_apply_qk_rope(query, key, cos, sin)
    return Q @ K.transpose(-1,-2)


def do_sdpa_attn(
        query, 
        key, 
        value, 
        cos, 
        sin, 
        out_proj: torch.nn.Linear = None,
        query_down_proj: torch.nn.Parameter = None,
        key_down_proj: torch.nn.Parameter = None,
        mask: torch.Tensor = None):
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)

    if query_down_proj is not None:
        Q = Q.transpose(1,2).flatten(2) @ query_down_proj
        Q = Q.unflatten(-1, (num_heads, -1)).transpose(1,2)

    if key_down_proj is not None:
        K = K.transpose(1,2).flatten(2) @ key_down_proj
        K = K.unflatten(-1, (num_heads, -1)).transpose(1,2)

    basic_mask = generate_mask(num_query, key.shape[-2], dtype=Q.dtype, device=Q.device)
    mask = torch.minimum(mask, basic_mask) if mask is not None else basic_mask

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query=Q,
        key=K,
        value=V,
        is_causal=False,
        attn_mask=mask)
    
    attn_output = attn_output.transpose(1,2).flatten(2)
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output
