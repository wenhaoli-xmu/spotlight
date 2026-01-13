import torch
import types
from .modify_llama import do_sdpa_attn, check_and_apply_qk_rope
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, CrossEntropyLoss
from ..modifier import Modifier
from peft import get_peft_model, LoraConfig, TaskType

from typing import List, Tuple
import json
from profiler import WallTime


def get_attn_score_using_angle_lsh(query, key, hash_fn, cos, sin):
    query, key = check_and_apply_qk_rope(query, key, cos, sin)

    q_inner = hash_fn(query)
    k_inner = hash_fn(key)

    q_hash = torch.sign(q_inner)
    k_hash = torch.sign(k_inner)

    sim = q_hash @ k_hash.transpose(-1,-2)
    return sim


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs
):
    # model forward function
    hidden_states, kv_cache = self.model(
        input_ids=input_ids,
        kv_cache=kv_cache)
    
    logits = self.lm_head(hidden_states[..., -1:,:]).float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    return CausalLMOutputWithPast(
        loss=loss, 
        logits=logits, 
        past_key_values=kv_cache)


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_cache is None:
        kv_cache = [None] * len(self.layers)

    for layer_idx, (decoder_layer, kv_cache_layer) in enumerate(zip(self.layers, kv_cache)):
        layer_output = decoder_layer(
            hidden_states, 
            kv_cache_layer)

        hidden_states, kv_cache_layer = layer_output
        kv_cache[layer_idx] = kv_cache_layer

    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None
):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)

    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, kv_cache = self.self_attn(
        hidden_states, 
        kv_cache)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads

    prefill_cond1 = hidden_states.shape[-2] > 1
    prefill_cond2 = kv_cache is None
    is_prefill = prefill_cond1 and prefill_cond2

    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)

    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    if kv_cache is not None:
        key_cache, val_cache = kv_cache
        keys = torch.cat([key_cache, keys], dim=-2)
        vals = torch.cat([val_cache, vals], dim=-2)

    kv_cache = (keys.data, vals.data)

    assert hidden_states.shape[0] == 1
    pos_ids = torch.arange(8192, dtype=torch.int64, device='cuda').unsqueeze_(0)
    cos, sin = self.rotary_emb(vals, position_ids=pos_ids)
    cos, sin = cos.squeeze(0), sin.squeeze(0)

    cond1 = self.draft_kwargs['enable'] is True
    cond2 = not self.is_fix_layer
    cond3 = not is_prefill  # pre-filling阶段不使用draft attention

    if cond1 and cond2 and cond3:

        draft_score = get_attn_score_using_angle_lsh(
            query=ques, 
            key=keys, 
            hash_fn=self.hash_fn, 
            cos=cos, 
            sin=sin)

        # 1. compute the topk indices
        def aggregate_topk(x, k):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            _, x_topk = x.topk(k=k, dim=-1)
            return x_topk

        num_kv_pair = draft_score.shape[-1]
        num_remain = num_kv_pair - int(num_kv_pair * self.draft_kwargs['mask_out'])
        num_remain = max(min(num_kv_pair, self.draft_kwargs['min_remain']), num_remain)
        draft_indices = aggregate_topk(draft_score, num_remain)

        # =========================================================================================================
        # NOTE: test
        # diff, loss = compute_attn_supervise_loss(draft_score, true_score, max_top=None, max_oth=1024, maskout=0.98)
        # print(f"layer-{self.layer_idx}: {diff}, {loss}")
        # =========================================================================================================


        # ==================================================================================
        # NOTE: test
        # if self.draft_kwargs['bench_mark']:

        #     # 2.5 run benchmark to evaluate the performance of draft strategy
        #     true_score = get_attn_score(query=ques, key=keys, cos=cos, sin=sin)
        #     true_indices = aggregate_topk(true_score, num_remain)
        #     self.ratios = []

        #     for draft_head, true_head in zip(draft_indices[0], true_indices[0]):
        #         ratios = []

        #         for qid, (draft_query, true_query) in enumerate(zip(draft_head, true_head)):
        #             draft_set = set(draft_query[:qid + 1].tolist())
        #             true_set = set(true_query[:qid + 1].tolist())

        #             intersect = draft_set.intersection(true_set)
        #             union = draft_set.union(true_set)
        #             ratio = len(intersect) / len(union)
        #             ratios.append(ratio)
                
        #         self.ratios.append(sum(ratios) / len(ratios))
        # ==================================================================================


        # 3. discard the unimportant token while keep the important 
        mask = torch.full(
            (1, num_heads, ques.shape[-2], num_kv_pair), 
            fill_value=torch.finfo(draft_score.dtype).min, 
            dtype=draft_score.dtype, 
            device=draft_score.device)
        mask = mask.scatter_(dim=-1, index=draft_indices, value=0)


        attn_output = do_sdpa_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            mask=mask,
            out_proj=self.o_proj)

    else:
        attn_output = do_sdpa_attn(
            query=ques,
            key=keys,
            value=vals,
            cos=cos,
            sin=sin,
            out_proj=self.o_proj)

    return attn_output, kv_cache


class MLPLayer(torch.nn.Module):
    def __init__(self, info, num_heads, in_features, out_features, silu=False, residue=True):
        super().__init__()
        get_init_value = lambda: torch.randn((1,num_heads,in_features,out_features), **info) * 0.001
        self.proj = torch.nn.Parameter(get_init_value(), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1,num_heads,1,out_features), **info), requires_grad=True)
        self.silu = torch.nn.SiLU() if silu else torch.nn.Identity()
        self.resi = residue

    def forward(self, x):
        dx = self.silu(x @ self.proj + self.bias)
        return dx + x if self.resi else dx


class MLPHashingFunction(torch.nn.Module):
    def __init__(self, info, num_heads, num_mlp_layers, mlp_dims, mlp_residue):
        super().__init__()
        mlp = torch.nn.ModuleList()
        for i in range(num_mlp_layers):
            mlp.append(MLPLayer(info, num_heads, mlp_dims[i], mlp_dims[i+1], i < num_mlp_layers - 1, mlp_residue))
        self.mlp = mlp
        

    def forward(self, x):
        for module in self.mlp:
            x = module(x)
        return x


class Decoder(torch.nn.Module):
    def _init_lora(
            self,
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float):

        target_modules = r".*\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules)
        self.decoder = get_peft_model(self.decoder, peft_config)


    @property
    def layers(self):
        if self.enable_lora:
            return self.decoder.base_model.model.model.layers
        else:
            return self.decoder.model.layers


    @property
    def model(self):
        if self.enable_lora:
            return self.decoder.base_model.model
        else:
            return self.decoder


    def reset(self):
        for layer in self.layers:
            if hasattr(layer.self_attn, 'k_cache'):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache


    def __init__(self, decoder, draft_kwargs):

        super().__init__()
        self.decoder = decoder
        self.enable_lora = False

        fix_layers = draft_kwargs.get('fix_layers', [])
        num_mlp_layers = draft_kwargs.get('num_mlp_layers', 2)
        mlp_dims = draft_kwargs.get('mlp_dims', '[128,128,128]')
        mlp_residue = draft_kwargs.get('mlp_residue', True)

        mlp_dims = json.loads(mlp_dims)

        self.fix_layers = fix_layers
        self.draft_kwargs = draft_kwargs

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer_idx, layer in enumerate(self.layers):

            info = {
                "device": layer.self_attn.q_proj.weight.data.device,
                "dtype": layer.self_attn.q_proj.weight.data.dtype}
            
            layer.self_attn.is_fix_layer = layer_idx in fix_layers

            # modify the forward function
            layer.self_attn.draft_kwargs = draft_kwargs
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.self_attn.hash_fn = MLPHashingFunction(info, self.model.config.num_attention_heads, num_mlp_layers, mlp_dims, mlp_residue)


    def is_benchmark_mode(self):
        return self.draft_kwargs['bench_mark']

    
    def enable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = True

    
    def disable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = False


    def get_ratios(self, reset=False):
        ratios = []
        for layer in self.layers:
            if layer.self_attn.is_fix_layer:
                ratios.append(None)
            else:
                ratios.append(layer.self_attn.ratios)
                del layer.self_attn.ratios
        return ratios
    

    def layer_ft_params(self, layer):
        layer = self.layers[layer]
        if layer.self_attn.is_fix_layer:
            return []
        return list(layer.self_attn.hash_fn.parameters())


    def ft_params(self, layer=None):
        params = []

        for layer in self.layers:
            params += layer.self_attn.hash_fn.parameters()

        return list(params)


    def forward(
            self, 
            input_ids, 
            labels=None,
            kv_cache=None):

        # decoder forward
        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels,
            kv_cache=kv_cache)

        return outputs


class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder
        ):
        super().__init__()
        self.decoder = decoder

    def ft_params(self):
        params = self.decoder.ft_params()
        return params


    def reset(self):
        self.decoder.reset()


    def forward(
            self,
            input_ids,
            kv_cache=None,
            labels=None,
            local_rank=None,
            **kwargs
        ):

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
            labels = torch.tensor(labels, dtype=torch.int64)[None, :]

        label_exist = labels is not None
        rank_exist = local_rank is not None

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if label_exist and labels.ndim == 3:
            labels = labels.flatten(0,1)

        if rank_exist:
            device = torch.device(local_rank)
        else:
            device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)

        outputs = self.decoder(
            input_ids, 
            labels=labels,
            kv_cache=kv_cache)

        return outputs


class Spotlight(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        draft_kwargs = self.conf['draft_kwargs']
        
        decoder = Decoder(model, draft_kwargs=draft_kwargs)
        decoder = Model(decoder)
        super().__init__(decoder, save_ckp, load_ckp)


    def ft_params(self):
        return self.model.ft_params()


    def reset(self):
        self.model.reset()


    def is_benchmark_mode(self):
        return self.model.decoder.is_benchmark_mode()

    
    def enable_benchmark_mode(self):
        return self.model.decoder.enable_benchmark_mode()

    
    def disable_benchmark_mode(self):
        return self.model.decoder.disable_benchmark_mode()
        

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2]):
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        output = self.model(input_ids=input_ids)
        logits = output.logits
        kv_cache = output.past_key_values 

        next_token_logits = logits[:, -1, :] 
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1) # 保持形状 [Batch, 1]
        
        new_ids = [next_token]

        for _ in range(max_new_tokens - 1):
            output = self.model(input_ids=next_token, past_key_values=kv_cache)
            logits, kv_cache = output.logits, output.past_key_values

            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1) # [Batch, 1]
            
            if input_ids.shape[0] == 1:
                if next_token.item() in eos_token_id:
                    break
            
            new_ids.append(next_token)

        if hasattr(self.model, 'reset'):
            self.model.reset()

        generated_sequence = torch.cat(new_ids, dim=1)
        
        return torch.cat([input_ids, generated_sequence], dim=1)
