import torch
import types
from typing import Optional

from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs, 
    Unpack,
    Cache,
    check_model_inputs,
    BaseModelOutputWithPast,
    TransformersKwargs,
    create_causal_mask,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen3,
    create_sliding_window_causal_mask,
    Qwen3Model
)

from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssModel,
    MoeModelOutputWithPast,
    apply_rotary_pos_emb as apply_rotary_pos_emb_oss,
)

def aggregate_topk(x, k):
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    _, x_topk = x.topk(k=k, dim=-1)
    return x_topk


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

    is_gpt_oss = isinstance(self, GptOssModel)
    is_qwen3 = isinstance(self, Qwen3Model)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if past_key_values is None or not isinstance(past_key_values, DynamicCache):
        past_key_values = DynamicCache()

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

        if is_qwen3:
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if hasattr(self, 'has_sliding_layers') and self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        elif is_gpt_oss:
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)
            }
        else: raise NotImplementedError(type(self))

    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs)

    hidden_states = self.norm(hidden_states)

    if is_qwen3:
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,)
    elif is_gpt_oss:
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values)
    else: raise NotImplementedError(type(self))


def topk_attention(query, key, value, budget):
    token_budget = int(budget * key.shape[2])

    group_size = query.shape[1] // key.shape[1]
    score = []
    for hidx in range(key.shape[1]):
        key_head = key[:, hidx: hidx+1]
        query_head = query[:, hidx * group_size: hidx * group_size + group_size]
        score_head = query_head @ key_head.transpose(2,3)
        score.append(score_head)
    score = torch.cat(score, dim=1)

    indices = torch.topk(score, k=token_budget, dim=-1).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=1)
    
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        is_causal=False,
        enable_gqa=True,
        ).transpose(1,2)


def flash_attention(query, key, value):
    assert query.shape[2] == key.shape[2]
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        enable_gqa=True,
        ).transpose(1,2)


def decoding_attention(query, key, value):
    assert query.shape[2] == 1
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=False,
        enable_gqa=True,
        ).transpose(1,2)


def attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    assert self.sliding_window is None, f"we do not support sliding window now."

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    cos, sin = position_embeddings

    if hasattr(self, 'q_norm'):
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1,2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)
        apply_rotary_pos_emb = apply_rotary_pos_emb_qwen3
    else:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1,2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)
        apply_rotary_pos_emb = apply_rotary_pos_emb_oss

    query_states, key_states = apply_rotary_pos_emb(
        query_states, 
        key_states, 
        cos, sin)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states, 
            value_states, 
            self.layer_idx)

    cond1 = self.draft_kwargs['enable'] is True
    cond2 = self.layer_idx not in self.draft_kwargs['fix_layers']
    budget = 1 - self.draft_kwargs['mask_out']
    
    decoding_stage = hidden_states.shape[1] == 1

    if cond1 and cond2 and decoding_stage:
        attn_output = topk_attention(
            query_states,
            key_states,
            value_states,
            budget)

    elif decoding_stage:
        attn_output = decoding_attention(
            query_states,
            key_states,
            value_states)
    else:
        attn_output = flash_attention(
            query_states,
            key_states,
            value_states)

    attn_output = attn_output.flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def monkey_patch(model, config):
    model.model.forward = types.MethodType(model_forward, model.model)
    for layer in model.model.layers:
        if layer.self_attn.sliding_window is None:
            layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
            layer.self_attn.draft_kwargs = config['draft_kwargs']
    return model
