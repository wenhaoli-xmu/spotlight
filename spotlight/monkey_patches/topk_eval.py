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


def topk_attention(query, key, value, position_embeddings, maskout, draft_kwargs):
    seq_len = key.shape[1]
    budget = int((1 - maskout) * seq_len)

    query = query.transpose(1,2)
    key = key.transpose(1,2)
    value = value.transpose(1,2)
    query, key = apply_rope(query, key, position_embeddings, True)

    rng = torch.arange(seq_len, dtype=torch.long, device=query.device)
    causal_mask = ~(rng[:, None] >= rng[None, :])[None, :, :]

    attn_output = []

    for head_idx in range(query.shape[1]):
        score_head = query[:, head_idx] @ key[:, head_idx].transpose(-1,-2)

        score_head = torch.masked_fill(score_head, causal_mask, torch.finfo(score_head.dtype).min)
        mask = torch.zeros_like(causal_mask)

        indices_head = torch.topk(score_head, k=budget, dim=-1).indices
        mask = mask.scatter(dim=-1, index=indices_head, value=1)
        mask = mask & (~causal_mask)

        cond = 'block_size' in draft_kwargs and draft_kwargs['block_size'] > 1

        if cond:
            """
            These lines simluate that all tokens within the block containing 
            the top-k tokens are selelcted for computation.
            """
            blk_size = int(draft_kwargs['block_size'])
            pad_len = int(math.ceil(seq_len / blk_size)) * blk_size - seq_len
            
            if pad_len > 0:
                mask_pad = torch.zeros(
                    (mask.shape[0], mask.shape[1], seq_len + pad_len), 
                    dtype=mask.dtype,
                    device=mask.device)
                mask_pad[..., :seq_len] = mask
            else:
                mask_pad = mask

            mask_pad = mask_pad.unflatten(-1, (-1,blk_size))
            mask_pad = torch.max(mask_pad, dim=-1, keepdim=True).values.expand(-1, -1, -1, blk_size).contiguous()
            mask_pad = mask_pad.flatten(-2)[..., :seq_len]
            mask = mask_pad & (~causal_mask)

        if draft_kwargs.get('plot_every', False):
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            plt.figure(figsize=[5,5], dpi=900)
            color_light = '#FBEBCF' 
            color_dark  = '#112641'
            cmap = ListedColormap([color_light, color_dark])
            plt.imshow(mask[0].cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            path = f"visualize/attn-blk-{draft_kwargs['block_size'] if cond else 1}.pdf"
            plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0)
            print(f"✅ Visualization saved to: {path}", flush=True)
            exit(0)

        out_head = torch.nn.functional.scaled_dot_product_attention(
            query[:, head_idx: head_idx + 1],
            key[:, head_idx: head_idx + 1],
            value[:, head_idx: head_idx + 1],
            attn_mask=mask.unsqueeze(1),
            is_causal=False)

        attn_output.append(out_head)

    attn_output = torch.cat(attn_output, dim=1)

    return attn_output.transpose(1,2)


def flash_attention(query, key, value, position_embeddings):
    assert query.shape[1] == key.shape[1]
    query_rope, key_rope = apply_rope(query, key, position_embeddings, head_first=False)
    return flash_attn_func(query_rope, key_rope, value, causal=True)


def decoding_attention(query, key, value, position_embeddings):
    assert query.shape[1] == 1

    query_rope, key_rope = apply_rope(query, key, position_embeddings, head_first=False)

    attn_output = flash_attn_func(
        query_rope,
        key_rope,
        value,
        causal=False)

    return attn_output


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

    cond1 = self.draft_kwargs['enable'] is True
    cond2 = self.layer_idx not in self.draft_kwargs['fix_layers']
    budget = self.draft_kwargs['mask_out']

    if cond1 and cond2:
        attn_output = topk_attention(
            query_states,
            key_states,
            value_states,
            position_embeddings,
            budget,
            self.draft_kwargs)
    else:
        attn_output = flash_attention(
            query_states,
            key_states,
            value_states,
            position_embeddings)

    attn_output = attn_output.flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def monkey_patch(model, config):
    model.model.forward = types.MethodType(model_forward, model.model)
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
        layer.self_attn.draft_kwargs = config['draft_kwargs']
    return model
