import torch
import types
from .modify_llama import do_sdpa_attn, generate_mask, get_attn_score, check_and_apply_qk_rope, segment
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from ..modifier import Modifier
from peft import get_peft_model, LoraConfig, TaskType

from typing import List, Tuple
import tqdm
import json


@torch.no_grad()
def log_diffs(true_attn, draft_attn, layer_idx):
    mask = torch.triu(torch.ones(true_attn.shape[-2:], dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
    true_attn = torch.masked_fill(true_attn, mask, value=torch.finfo(true_attn.dtype).min)
    indices = torch.argsort(true_attn, dim=-1, descending=True)

    top_cnt = int(indices.shape[-1] * 0.02)
    top_indices = indices[..., :top_cnt]
    oth_indices = indices[..., top_cnt:]

    top_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=top_indices)[..., :, None]
    oth_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=oth_indices)[..., None, :]

    top_draft_attn = torch.gather(draft_attn, dim=-1, index=top_indices)[..., :, None]
    oth_draft_attn = torch.gather(draft_attn, dim=-1, index=oth_indices)[..., None, :]

    total_diff = 0
    total_compare = 0

    for top_seg, oth_seg, top_mask_seg, oth_mask_seg in tqdm.tqdm(
        zip(
        segment(top_draft_attn, dim=1, n=1),
        segment(oth_draft_attn, dim=1, n=1),
        segment(top_mask, dim=1, n=1),
        segment(oth_mask, dim=1, n=1)),
        desc=f'layer-{layer_idx}'
    ):

        residual = (top_seg - oth_seg)
        residual_mask = (top_mask_seg | oth_mask_seg).expand_as(residual).flatten(-3)
        logits = residual.flatten(-3)[~residual_mask.bool()]

        diff = torch.count_nonzero(logits < 0)
        compare = logits.numel()

        total_diff += diff
        total_compare += compare

    # 算一下排序误差
    diff = total_diff / total_compare
    print(f"diff: {diff.item():<.3f}")


def get_attn_score_using_inner_prod(query, key, cos, sin):
    query, key = check_and_apply_qk_rope(query, key, cos, sin)
    sim = query @ key.transpose(-1,-2)
    return sim


def random_rotation_matrix(dim, dtype, device):
    """
    随机生成一个 n 维旋转矩阵
    :param dim: 维度大小 (n)
    :return: n x n 随机旋转矩阵
    """
    # 使用QR分解生成随机正交矩阵
    random_matrix = torch.randn((dim, dim), dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    
    # 调整使其行列式为1
    if torch.det(q) < 0:
        q[:, 0] *= -1

    return q.type(dtype).to(device)


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs
):
    # model forward function
    hidden_states, kv_cache, draft_attn, true_attn = self.model(
        input_ids=input_ids,
        kv_cache=kv_cache)
    
    logits = self.lm_head(hidden_states).cpu().float()

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
        past_key_values=kv_cache,
        attentions=(draft_attn, true_attn))


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_cache is None:
        kv_cache = [None] * len(self.layers)

    draft_attns = []
    true_attns = []

    for layer_idx, (decoder_layer, kv_cache_layer) in enumerate(zip(self.layers, kv_cache)):
        layer_output = decoder_layer(
            hidden_states, 
            kv_cache_layer)

        hidden_states, kv_cache_layer, draft_attn, true_attn = layer_output
        draft_attns.append(draft_attn)
        true_attns.append(true_attn)

        kv_cache[layer_idx] = kv_cache_layer

    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache, draft_attns, true_attns


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)

    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_cache, draft_attn, true_attn = self.self_attn(
        hidden_states, 
        kv_cache)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache, draft_attn, true_attn


def compute_attn_supervise_loss(draft_attn, true_attn, max_top, max_oth, maskout):
    loss = torch.tensor(0, dtype=torch.float32)
    diff = 0
    total = 0
    criterion = torch.nn.BCEWithLogitsLoss()
        
    # 计算出true attn的sort index
    mask = torch.triu(torch.ones(true_attn.shape[-2:], dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
    true_attn = torch.masked_fill(true_attn, mask, value=torch.finfo(true_attn.dtype).min)
    indices = torch.argsort(true_attn, dim=-1, descending=True)

    # 切分出来top 0.01 的indices，和other 0.98的indices
    top_cnt = int(indices.shape[-1] * (1 - maskout))
    top_indices = indices[..., :top_cnt]
    oth_indices = indices[..., top_cnt:]

    if max_top is not None:
        top_rnd_indices = torch.randperm(top_cnt, dtype=torch.int64, device=indices.device)[:max_top]
        top_indices = top_indices[..., top_rnd_indices]
    if max_oth is not None:
        oth_rnd_indices = torch.randperm(indices.shape[-1] - top_cnt, dtype=torch.int64, device=indices.device)[:max_oth]
        oth_indices = oth_indices[..., oth_rnd_indices]

    top_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=top_indices)[..., :, None]
    oth_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=oth_indices)[..., None, :]

    top_draft_attn = torch.gather(draft_attn, dim=-1, index=top_indices)[..., :, None]
    oth_draft_attn = torch.gather(draft_attn, dim=-1, index=oth_indices)[..., None, :]

    num_heads = top_draft_attn.shape[1]

    for top_head, oth_head, top_mask_head, oth_mask_head in zip(
        segment(top_draft_attn, dim=1, n=1),
        segment(oth_draft_attn, dim=1, n=1),
        segment(top_mask, dim=1, n=1),
        segment(oth_mask, dim=1, n=1)
    ):

        residual = top_head - oth_head
        residual_mask = (top_mask_head | oth_mask_head).expand_as(residual).flatten(-3)

        logits = residual.flatten(-3)[~residual_mask.bool()]
        labels = torch.ones_like(logits, dtype=torch.float32)
        loss += criterion(logits, labels.type(torch.float32)).cpu()
        
        diff += torch.count_nonzero(logits < 0).item()
        total += logits.numel()

    diff /= total
    loss /= num_heads

    return diff, loss


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
    ret_attn = (None, None)

    assert hidden_states.shape[0] == 1
    pos_ids = torch.arange(8192, dtype=torch.int64, device='cuda').unsqueeze_(0)
    cos, sin = self.rotary_emb(vals, position_ids=pos_ids)
    cos, sin = cos.squeeze(0), sin.squeeze(0)

    cond1 = self.draft_kwargs['enable'] is True
    cond2 = not self.is_fix_layer

    if cond1 and cond2:

        draft_score = get_attn_score_using_inner_prod(ques, keys, cos, sin)

        # pre-filling stage should do causal attention
        if is_prefill:
            mask = generate_mask(*draft_score.shape[-2:], dtype=draft_score.dtype, device=draft_score.device)
            draft_score += mask

        # 2. compute the topk indices
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


        # =========================================================================================================
        # NOTE: test
        if self.draft_kwargs['bench_mark']:
            # 2.5 run benchmark to evaluate the performance of draft strategy
            true_score = get_attn_score(query=ques, key=keys, cos=cos, sin=sin)
            true_indices = aggregate_topk(true_score, num_remain)
            self.ratios = []

            for draft_head, true_head in zip(draft_indices[0,:,-1,:], true_indices[0,:,-1,:]):
                draft_set = set(draft_head.tolist())
                true_set = set(true_head.tolist())

                intersect = draft_set.intersection(true_set)
                union = draft_set.union(true_set)
                ratio = len(intersect) / len(union)
                self.ratios.append(ratio)
        # =========================================================================================================


        # 3. discard the unimportant token while keep the important 
        mask = torch.full(
            (1, num_heads, ques.shape[-2], num_kv_pair), 
            fill_value=torch.finfo(draft_score.dtype).min, 
            dtype=draft_score.dtype, 
            device=draft_score.device)
        mask = mask.scatter_(dim=-1, index=draft_indices, value=0)

        query_idx = torch.arange(mask.shape[2], device=mask.device)
        key_idx = torch.arange(mask.shape[3], device=mask.device)
        causal_cond = query_idx[:, None] < key_idx[None, :]
        causal_cond = causal_cond[None, None, :, :].expand_as(mask).to(mask.device)
        mask = torch.where(causal_cond, torch.finfo(mask.dtype).min, mask)

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

    return attn_output, kv_cache, *ret_attn


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


    def __init__(
            self, 
            decoder, 
            enable_lora: bool = False,
            lora_kwargs: dict = None,
            fix_layers: list = [],
            draft_kwargs: dict = {"use_draft": False}):

        super().__init__()
        self.decoder = decoder
        self.enable_lora = False

        fix_layers = draft_kwargs.get('fix_layers', [])
        self.fix_layers = fix_layers
        self.draft_kwargs = draft_kwargs

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.is_fix_layer = layer_idx in fix_layers
            layer.self_attn.draft_kwargs = draft_kwargs
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)


    def is_benchmark_mode(self):
        return self.draft_kwargs['bench_mark']

    
    def enable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = True

    
    def disable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = False


    def get_ratios(self, reset=False):
        ratios = []
        for idx, layer in enumerate(self.layers):
            if idx in self.fix_layers:
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
            if not layer.self_attn.is_fix_layer:
                params += layer.self_attn.hash_fn.parameters()

        return list(params)


    def forward(
            self, 
            input_ids, 
            labels=None):

        # decoder forward
        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels)

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
            labels=labels)

        return outputs


class UpperBound(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]

        draft_kwargs = self.conf['draft_kwargs']
        fix_layers = [] if "fix_layers" not in self.conf else self.conf["fix_layers"]
        
        decoder = Decoder(
            model, 
            enable_lora=enable_lora,
            lora_kwargs=lora_kwargs,
            fix_layers=fix_layers,
            draft_kwargs=draft_kwargs)

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
            input_ids = input_ids.flatten(0,1)

        # put the tensor on to the model's device
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        # prefilling
        prefill_ids = input_ids[:, :-1]
        self.model(input_ids=prefill_ids)

        # generation
        new_tok = input_ids[:, -1:]
        new_ids = []
        while len(new_ids) < max_new_tokens:
            logits = self.model(input_ids=new_tok).logits
            new_tok = logits.argmax(dim=-1)
            if new_tok.ravel().item() in eos_token_id: break
            new_ids.append(new_tok.ravel().item())

        self.model.reset()
        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
        return torch.cat([input_ids, new_ids], dim=-1)
