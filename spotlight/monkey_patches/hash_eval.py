import torch
import types
from typing import Optional, Union, Any
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

from .utils.hash_utils import HashModule
from dataclasses import dataclass
import tqdm
from pygments.console import colorize
from itertools import chain


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


@dataclass
class CausalLMOutputWithMetrics(CausalLMOutputWithPast):
    metrics: Optional[tuple[dict]] = None


@torch.no_grad()
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
    outputs, metrics = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithMetrics(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        metrics=metrics
    )


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
    metrics = []

    for decoder_layer in tqdm.tqdm(self.layers[: self.config.num_hidden_layers]):
        hidden_states, metric = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=self.position_embeddings,
            **kwargs)
        metrics.append(metric)


    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None), metrics


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

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    query_states, key_states = apply_rope(query_states, key_states, position_embeddings, False)

    metric = dict()

    if hasattr(self, 'query_hash'):
        q_hash = self.query_hash(query_states) > 0
        k_hash = self.key_hash(key_states) > 0
        group_size = q_hash.shape[2] // k_hash.shape[2]

        attn_output = []
        num_select = []
        precision = [] # q
        recall = [] # q

        est_mask = []
        oracle_mask = []

        chunk_size = 1024
        for chunk_start in range(0, q_hash.shape[1], chunk_size):
            chunk_end = min(q_hash.shape[1], chunk_start + chunk_size)
            q_hash_chunk = q_hash[:, chunk_start: chunk_end]
            k_hash_chunk = k_hash[:, :chunk_end]

            q_indices = torch.arange(chunk_start, chunk_end, device='cuda')[:, None]
            k_indices = torch.arange(0, chunk_end, device='cuda')[None, :]
            causal_mask = q_indices >= k_indices

            q_f = q_hash_chunk.float()
            k_f = k_hash_chunk.repeat_interleave(group_size, 2).float()

            q_norm = q_f.sum(dim=-1).unsqueeze(2) 
            k_norm = k_f.sum(dim=-1).unsqueeze(1)
            dot_prod = torch.einsum('bihd,bjhd->bijh', q_f, k_f)

            sim = q_norm + k_norm - 2 * dot_prod
            sim = sim.permute(0, 3, 1, 2)
            select_mask = sim < int(self.head_dim * self.hash_config.get("select_thresh", 0.5))

            mask = causal_mask & select_mask
            num_select.append(mask.sum(-1))
            
            est_mask.append(mask[0,0].cpu())

            q_chunk = query_states[:, chunk_start: chunk_end]
            k_chunk = key_states[:, :chunk_end]
            v_chunk = value_states[:, :chunk_end]

            q_head_chunk = q_chunk[..., 0, :]
            k_head_chunk = k_chunk[..., 0, :]

            for q_idx in range(mask.shape[2]):

                kv_length = chunk_start + q_idx + 1
                topk = int(self.hash_config['topk'] * kv_length)
                topk = max(1, topk)

                _query = q_head_chunk[0, q_idx] # (d,)
                _key = k_head_chunk[0, :kv_length] # (nk, d)

                label_indices = torch.einsum("d,kd->k", _query, _key).topk(dim=-1, k=topk).indices.tolist()

                _oracle_mask = torch.zeros((kv_length,), dtype=torch.bool, device='cpu')
                _oracle_mask.scatter_(dim=-1, index=torch.tensor(label_indices, device='cpu'), value=True)
                oracle_mask.append(_oracle_mask)
                
                current_mask = mask[0, 0, q_idx, :kv_length]
                head_predict = set(torch.nonzero(current_mask).ravel().tolist())

                label_sets = set(label_indices)
                true_pos = len(head_predict.intersection(label_sets))
                
                len_predict = len(head_predict)
                _precision = true_pos / len_predict if len_predict > 0 else 0.0
                
                len_label = len(label_sets)
                _recall = true_pos / len_label if len_label > 0 else 0.0

                precision.append(_precision)
                recall.append(_recall)

            attn_output_chunk = torch.nn.functional.scaled_dot_product_attention(
                q_chunk.transpose(1,2),
                k_chunk.transpose(1,2),
                v_chunk.transpose(1,2),
                is_causal=False,
                attn_mask=mask,
                enable_gqa=True)

            attn_output.append(attn_output_chunk)

        attn_output = torch.cat(attn_output, dim=2)
        num_select = torch.cat(num_select, dim=2)
        precision = torch.tensor(precision)
        recall = torch.tensor(recall)

        metric.update({
            "num_tokens": hidden_states.shape[1],
            "num_select": num_select,
            "precision": precision,
            "recall": recall})
        
        if self.hash_config.get('visualize_head0', False):
            thresh_prefix = f"thresh_{self.hash_config.get('select_thresh', 0.5)}"
            topk_prefix = f"topk_{self.hash_config.get('topk', 0.05)}"
            os.makedirs(f"visualize/{thresh_prefix}", exist_ok=True)
            os.makedirs(f"visualize/{topk_prefix}", exist_ok=True)
            seqlen = hidden_states.shape[1]
            
            full_est = torch.zeros((seqlen, seqlen), dtype=torch.float32)
            full_oracle = torch.zeros((seqlen, seqlen), dtype=torch.float32)
            
            for i, m_est in enumerate(est_mask):
                start_row = i * chunk_size
                end_row = min(start_row + chunk_size, seqlen)
                _, cols = m_est.shape
                full_est[start_row:end_row, :cols] = m_est
            
            for i, m_oracle in enumerate(oracle_mask):
                cols = m_oracle.shape[0]
                full_oracle[i, :cols] = m_oracle
                
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            
            color_light = '#FBEBCF' 
            color_dark  = '#112641'
            cmap = ListedColormap([color_light, color_dark])

            def save_single_plot(matrix_tensor, prefix):
                plt.figure(figsize=[5, 5], dpi=300)
                
                plt.imshow(matrix_tensor.numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                plt.axis('off')
                plt.tight_layout()
                
                path = f"visualize/{prefix}/layer-{self.layer_idx}.jpg"
                plt.savefig(path, bbox_inches='tight', pad_inches=0)
                
                plt.close() 
                return path

            path_est = save_single_plot(full_est, thresh_prefix)
            path_oracle = save_single_plot(full_oracle, topk_prefix)

            print(f"✅ Visualization saved to:\n  - {path_est}\n  - {path_oracle}", flush=True)

    else:

        query_states = query_states.transpose(1,2)
        key_states = key_states.transpose(1,2)
        value_states = value_states.transpose(1,2)
            
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            enable_gqa=True)

    attn_output = attn_output.transpose(1,2).flatten(2).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, metric


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
    
    hidden_states, metrics = self.self_attn(
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
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, metrics


def monkey_patch(model, config):
    model.forward = types.MethodType(causal_forward, model)
    model.model.forward = types.MethodType(model_forward, model.model)

    model_name = model.config._name_or_path.split('/')[-1].lower()
    load_dir = f"train_results/{model_name}-bkp"

    for layer_idx, layer in enumerate(model.model.layers):
        layer.forward = types.MethodType(layer_forward, layer)
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)

        if layer_idx not in config['freeze_layers']:
            layer.self_attn.hash_config = config
            weight_path = os.path.join(load_dir, f"weight-{layer_idx}.pth")
            
            if os.path.exists(weight_path):
                weights = torch.load(weight_path, map_location='cuda')

                layer.self_attn.key_hash = HashModule(
                    layer.self_attn.config.num_key_value_heads,
                    layer.self_attn.head_dim,
                    n_layers=config['num_hash_layers'],
                    dropout=0.0)

                layer.self_attn.query_hash = HashModule(
                    layer.self_attn.config.num_attention_heads,
                    layer.self_attn.head_dim,
                    n_layers=config['num_hash_layers'],
                    dropout=0.0)

                params = chain.from_iterable((
                    layer.self_attn.query_hash.parameters(), 
                    layer.self_attn.key_hash.parameters()))

                for p, w in zip(params, weights):
                    p.data = w

            elif not config['ignore_missing_weights']:
                raise RuntimeError(f"Missing checkpoint {weight_path}.")

            else:
                print(colorize("yellow", f"Missing checkpoint {weight_path}"))

    return model
