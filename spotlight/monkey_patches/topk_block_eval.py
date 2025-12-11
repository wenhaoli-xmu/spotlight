import math
import torch
import types
from flash_attn import flash_attn_func
from typing import Optional, Any, Iterable

from transformers.cache_utils import DynamicLayer
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs, 
    Unpack,
    Cache,
    check_model_inputs,
    BaseModelOutputWithPast,
    TransformersKwargs,
    Cache,
    create_causal_mask,
    create_sliding_window_causal_mask,
)


def maybe_pad_query(x, tile):
    if x.shape[2] % tile != 0:
        remain = tile - x.shape[2] % tile
        x_pad = torch.zeros(
            (x.shape[0], x.shape[1], remain, x.shape[3]), 
            dtype=x.dtype, device=x.device)
        x = torch.cat([x, x_pad], dim=2)
    return x


def aggregate_topk(x, k):
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    _, x_topk = x.topk(k=k, dim=-1)
    return x_topk


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, head_dim=2):
    cos = cos.unsqueeze(head_dim)
    sin = sin.unsqueeze(head_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


@check_model_inputs
def model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_key_values = None

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    if not hasattr(self, 'position_embeddings'):
        position_ids = torch.arange(self.config.max_position_embeddings, dtype=torch.long, device='cuda')
        position_ids = position_ids.unsqueeze(0)
        self.position_embeddings = self.rotary_emb(hidden_states, position_ids)
    position_ids = None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=self.position_embeddings,
            **kwargs)

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,)


def apply_rope(query, key, position_embeddings, head_first=True):
    cos, sin = position_embeddings
    head_dim = 1 if head_first else 2
    seq_dim = 2 if head_first else 1

    query_length = query.shape[seq_dim]
    key_length = key.shape[seq_dim]
    
    cos = cos[:, :key_length]
    sin = sin[:, :key_length]

    if query_length == key_length:
        query = apply_rotary_pos_emb(query, cos, sin, head_dim=head_dim)
    else:
        query = apply_rotary_pos_emb(
            query, 
            cos[:, -query_length:], 
            sin[:, -query_length:], 
            head_dim=head_dim)

    key = apply_rotary_pos_emb(key, cos, sin, head_dim=head_dim)

    return query, key


def topk_block_attention(query, key, value, position_embeddings, block_kwargs):
    seq_len = key.shape[1]
    blk_len = int(math.ceil(seq_len / block_kwargs['block_size']))

    blk_rng = torch.arange(blk_len, dtype=torch.long, device=query.device)
    blk_causal_mask = ~(blk_rng[:, None] >= blk_rng[None, :])[None, None, :, :]

    rng = torch.arange(seq_len, dtype=torch.long, device=query.device)
    causal_mask = ~(rng[:, None] >= rng[None, :])[None, None, :, :]

    query = query.transpose(1,2)
    key = key.transpose(1,2)
    value = value.transpose(1,2)

    # ============
    # Mean Pooling
    # ============

    accum_score = None

    for h_idx in range(query.shape[1]):
        group_size = query.shape[1] // key.shape[1]
        kv_group_id = h_idx // group_size
        query_head = query[:, h_idx: h_idx + 1]
        key_head = key[:, kv_group_id: kv_group_id + 1]

        key_mean = key_head.mean(2, keepdim=True).expand(-1, -1, seq_len, -1).contiguous()
        query_mean = query_head.mean(2, keepdim=True).expand(-1, -1, seq_len, -1).contiguous()
        query_mean, key_mean = apply_rope(query_mean, key_mean, position_embeddings, True)

        mean_score = query_mean @ key_mean.transpose(-1,-2) / query_mean.shape[-1] ** 0.5
        mean_score.masked_fill_(causal_mask.expand(-1, mean_score.shape[1], -1, -1), torch.finfo(mean_score.dtype).min)
        mean_score = mean_score.softmax(dim=-1)

        if accum_score is None:
            accum_score = mean_score
        else:
            accum_score += mean_score

    mean_score = accum_score.squeeze(0,1)
    mean_score_topk_idx = mean_score.topk(k=64 * block_kwargs['block_size'], dim=-1, sorted=True).indices
    mean_score_topk_mask = torch.zeros_like(mean_score, dtype=torch.bool).scatter(dim=-1, index=mean_score_topk_idx, value=True)
    mean_score_topk_mask = mean_score_topk_mask & (~causal_mask[0,0])

    # ===========
    # Max Pooling
    # ===========
    
    accum_score = None

    for h_idx in range(query.shape[1]):
        group_size = query.shape[1] // key.shape[1]
        kv_group_id = h_idx // group_size
        query_head = query[:, h_idx: h_idx + 1]
        key_head = key[:, kv_group_id: kv_group_id + 1]

        key_max = key_head.max(2, keepdim=True).values.expand(-1, -1, seq_len, -1).contiguous()
        query_max = query_head.max(2, keepdim=True).values.expand(-1, -1, seq_len, -1).contiguous()
        query_max, key_max = apply_rope(query_max, key_max, position_embeddings, True)

        max_score = query_max @ key_max.transpose(-1,-2) / query_max.shape[-1] ** 0.5
        max_score.masked_fill_(causal_mask.expand(-1, max_score.shape[1], -1, -1), torch.finfo(max_score.dtype).min)
        max_score = max_score.softmax(dim=-1)

        if accum_score is None:
            accum_score = max_score
        else:
            accum_score += max_score

    max_score = accum_score.squeeze(0,1)
    max_score = max_score.unflatten(-2, (-1, block_kwargs['block_size'])).sum(-2).unflatten(-1, (-1, block_kwargs['block_size'])).sum(-1)
    max_score_topk_idx = max_score.topk(k=64, dim=-1, sorted=True).indices
    max_score_topk_mask = torch.zeros_like(max_score, dtype=torch.bool).scatter(dim=-1, index=max_score_topk_idx, value=True)
    max_score_topk_mask = max_score_topk_mask & (~blk_causal_mask[0,0])

    # =============
    # Topk w/o RoPE
    # =============

    accum_score = None

    for h_idx in range(query.shape[1]):
        group_size = query.shape[1] // key.shape[1]
        kv_group_id = h_idx // group_size
        query_head = query[:, h_idx: h_idx + 1]
        key_head = key[:, kv_group_id: kv_group_id + 1]

        head_score = query_head @ key_head.transpose(-1,-2) / query_head.shape[-1] ** 0.5
        head_score.masked_fill_(causal_mask.expand(-1, head_score.shape[1], -1, -1), torch.finfo(head_score.dtype).min)
        head_score = head_score.softmax(dim=-1)

        if accum_score is None:
            accum_score = head_score
        else:
            accum_score += head_score

    topk_wo_rope_score = accum_score.squeeze(0,1)
    topk_wo_rope_score_idx = topk_wo_rope_score.topk(k=64 * block_kwargs['block_size'], dim=-1, sorted=True).indices
    topk_wo_rope_score_mask = torch.zeros_like(topk_wo_rope_score, dtype=torch.bool).scatter(dim=-1, index=topk_wo_rope_score_idx, value=True)
    topk_wo_rope_score_mask = topk_wo_rope_score_mask & (~causal_mask[0,0])

    query, key = apply_rope(query, key, position_embeddings, True)

    # =========
    # RoPE Only
    # =========

    query_pad = maybe_pad_query(query, block_kwargs['block_size'])
    query_avgs = query_pad.contiguous().unflatten(1, (key.shape[1], -1)).sum(2)
    query_avgs = query_avgs.unflatten(2, (-1, block_kwargs['block_size'])).mean(3)

    key_pad = maybe_pad_query(key, block_kwargs['block_size'])
    key_avgs = key_pad.contiguous().unflatten(2, (-1, block_kwargs['block_size'])).mean(3)

    avg_score = query_avgs @ key_avgs.transpose(-1,-2) / query_avgs.shape[-1] ** 0.5
    avg_score.masked_fill_(blk_causal_mask.expand(-1, avg_score.shape[1], -1, -1), torch.finfo(avg_score.dtype).min)
    avg_score = avg_score.softmax(dim=-1)
    avg_score_topk_idx = avg_score.sum([0,1]).topk(k=64, dim=-1, sorted=True).indices
    avg_score_topk_mask = torch.zeros_like(avg_score[0,0], dtype=torch.bool).scatter(dim=-1, index=avg_score_topk_idx, value=True)
    avg_score_topk_mask = avg_score_topk_mask & (~blk_causal_mask[0,0])

    # ===========
    # Oracle Topk
    # ===========

    accum_score = None

    for h_idx in range(query.shape[1]):
        group_size = query.shape[1] // key.shape[1]
        kv_group_id = h_idx // group_size
        query_head = query[:, h_idx: h_idx + 1]
        key_head = key[:, kv_group_id: kv_group_id + 1]

        head_score = query_head @ key_head.transpose(-1,-2) / query_head.shape[-1] ** 0.5
        head_score.masked_fill_(causal_mask.expand(-1, head_score.shape[1], -1, -1), torch.finfo(head_score.dtype).min)
        head_score = head_score.softmax(dim=-1)

        if accum_score is None:
            accum_score = head_score
        else:
            accum_score += head_score

    topk_score = accum_score.squeeze(0,1)
    topk_score_idx = topk_score.topk(k=64 * block_kwargs['block_size'], dim=-1, sorted=True).indices
    topk_score_mask = torch.zeros_like(topk_score, dtype=torch.bool).scatter(dim=-1, index=topk_score_idx, value=True)
    topk_score_mask = topk_score_mask & (~causal_mask[0,0])
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    plt.figure(figsize=[5,5], dpi=900)
    color_light = '#FBEBCF' 
    color_dark  = '#112641'
    cmap = ListedColormap([color_light, color_dark])
    plt.imshow(avg_score_topk_mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    path = f"visualize/avg-score-{block_kwargs['block_size']}.pdf"
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=[5,5], dpi=900)
    color_light = '#FBEBCF'
    color_dark  = '#112641'
    cmap = ListedColormap([color_light, color_dark])
    plt.imshow(mean_score_topk_mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    path = f"visualize/mean-score-{block_kwargs['block_size']}.pdf"
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=[5,5], dpi=900)
    color_light = '#FBEBCF'
    color_dark  = '#112641'
    cmap = ListedColormap([color_light, color_dark])
    plt.imshow(topk_score_mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    path = f"visualize/topk-score-{block_kwargs['block_size']}.pdf"
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=[5,5], dpi=900)
    color_light = '#FBEBCF'
    color_dark  = '#112641'
    cmap = ListedColormap([color_light, color_dark])
    plt.imshow(max_score_topk_mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    path = f"visualize/max-score-{block_kwargs['block_size']}.pdf"
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=[5,5], dpi=900)
    color_light = '#FBEBCF'
    color_dark  = '#112641'
    cmap = ListedColormap([color_light, color_dark])
    plt.imshow(topk_wo_rope_score_mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    path = f"visualize/topk-wo-rope-score-{block_kwargs['block_size']}.pdf"
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0)

    print(f"✅ Visualization saved to: {path}", flush=True)
    exit(0)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        enable_gqa=True)

    return attn_output.transpose(1,2)


def attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    assert self.sliding_window is None, f"we do not support sliding window currently."

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    num_group = query_states.shape[2] // key_states.shape[2]
    key_states = key_states.repeat_interleave(num_group, dim=2)
    value_states = value_states.repeat_interleave(num_group, dim=2)

    attn_output = topk_block_attention(
        query_states,
        key_states,
        value_states,
        position_embeddings,
        self.block_kwargs)

    attn_output = attn_output.flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def monkey_patch(model, config):
    model.model.forward = types.MethodType(model_forward, model.model)
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
        layer.self_attn.block_kwargs = config['block_kwargs']
    return model
