import torch
import types
from flash_attn import flash_attn_func
from typing import Optional, Union

from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs, 
    Unpack,
    Cache,
    check_model_inputs,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    TransformersKwargs,
    Cache,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from .utils.hash_utils import (
    HashModule, 
    compute_lsh_score,
    compute_attn_score)


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


def causal_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

    >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs)

    return CausalLMOutputWithPast(
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions)


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
    attentions = []

    return_hidden_states = kwargs.get('return_hidden_states', False)
    middle_hidden_states = [] if return_hidden_states else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states, ret_scores = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=self.position_embeddings,
            **kwargs)
        attentions.append(ret_scores)
        if return_hidden_states:
            middle_hidden_states.append(hidden_states)

    attentions = tuple(attentions)

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=middle_hidden_states,
        attentions=attentions)


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


def flash_attention(
        query, 
        key, 
        q_hash, 
        k_hash, 
        value, 
        position_embeddings, 
        return_hidden_states,
        random_query_index):

    assert query.shape[1] == key.shape[1]

    ret_scores = None
    if not return_hidden_states:
        hash_score = compute_lsh_score(q_hash, k_hash, random_query_index)
        attn_score = compute_attn_score(query, key, random_query_index)
        return None, (hash_score, attn_score)

    attn_output = flash_attn_func(query, key, value, causal=True)
    return attn_output, ret_scores


def attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    assert self.sliding_window is None, f"we do not support sliding window currently."

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    return_hidden_states = kwargs.get('return_hidden_states', False)
    random_query_index = kwargs.get("random_query_index", None)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))

    query_states, key_states = apply_rope(query_states, key_states, position_embeddings, head_first=False)

    value_states = None
    if return_hidden_states:
        value_states = self.v_proj(hidden_states).view(hidden_shape)

    q_hash = self.query_hash(query_states)
    k_hash = self.key_hash(key_states)

    attn_output, ret_scores = flash_attention(
        query_states,
        key_states,
        q_hash,
        k_hash,
        value_states,
        position_embeddings,
        return_hidden_states=return_hidden_states,
        random_query_index=random_query_index)

    if return_hidden_states:
        attn_output = attn_output.flatten(2).contiguous()
        attn_output = self.o_proj(attn_output)

    return attn_output, ret_scores


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    hidden_states, ret_scores = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    assert torch.is_grad_enabled() == False
    chunk_size = 4096
    outputs = []
    for start in range(0, hidden_states.shape[1], chunk_size):
        end = min(hidden_states.shape[1], start + chunk_size)
        chunk_states = hidden_states[:, start: end]
        outputs.append(self.mlp(chunk_states))
    hidden_states = torch.cat(outputs, dim=1)

    hidden_states = residual + hidden_states
    return hidden_states, ret_scores


def dump_as_attn_modules(self):
    attn_modules = []
    for layer in self.model.layers:
        attn_modules.append(layer.self_attn)
    return attn_modules


def monkey_patch(model, config):
    model.forward = types.MethodType(causal_forward, model)
    model.model.forward = types.MethodType(model_forward, model.model)
    model.dump_as_attn_modules = types.MethodType(dump_as_attn_modules, model)

    # layer_forward_compiled = torch.compile(layer_forward, dynamic=True)

    for layer in model.model.layers:

        layer.forward = types.MethodType(layer_forward, layer)
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)

        layer.self_attn.key_hash = HashModule(
            layer.self_attn.config.num_key_value_heads,
            layer.self_attn.head_dim,
            n_layers=config['num_hash_layers'],
            dropout=config['hash_dropout'])

        layer.self_attn.query_hash = HashModule(
            layer.self_attn.config.num_attention_heads,
            layer.self_attn.head_dim,
            n_layers=config['num_hash_layers'],
            dropout=config['hash_dropout'])

    return model
