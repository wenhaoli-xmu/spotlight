import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import attn_k8_q32 
import packbits
import lru_cache_kernel


def compute_lsh_score(q_hash, k_hash, random_query_index):
    group_size = q_hash.shape[2] // k_hash.shape[2]

    if random_query_index is not None:
        q_hash = q_hash[:, random_query_index]

    q_hash = q_hash.transpose(1, 2)
    k_hash = k_hash.transpose(1, 2)
    k_hash = k_hash.unsqueeze(2).expand(-1, -1, group_size, -1, -1).flatten(1,2)
    
    q_bin = (q_hash > 0).float() * 2 - 1
    k_bin = (k_hash > 0).float() * 2 - 1
    
    q_soft = torch.tanh(q_hash)
    k_soft = torch.tanh(k_hash)
    
    q_final = (q_bin - q_soft).detach() + q_soft
    k_final = (k_bin - k_soft).detach() + k_soft

    sim = q_final @ k_final.transpose(-1, -2) / q_final.shape[-1] ** 0.5
    
    return sim


def compute_attn_score(q, k, random_query_index):
    score = []
    num_heads = q.shape[2]
    group_size = num_heads // k.shape[2]

    if random_query_index is not None:
        q = q[:, random_query_index]

    rng = torch.arange(k.shape[1], device=q.device)
    msk = random_query_index[:, None] < rng[None, :]
    msk = msk[None, None, :, :]

    for head_idx in range(num_heads):
        q_head = q[..., head_idx, :]
        k_head = k[..., head_idx // group_size , :]
        head_score = q_head @ k_head.transpose(-1,-2) / q_head.shape[-1] ** 0.5
        head_score.masked_fill(msk, value=torch.finfo(head_score.dtype).min)
        head_score = head_score.softmax(dim=-1, dtype=torch.float).type(q.dtype)
        score.append(head_score)

    return torch.stack(score, dim=1)


def async_hashing_worker(x, hash_embed, buffer, n_tokens):
    embed = hash_embed(x)
    x_bins = packbits_kernel(embed)
    buffer[:, n_tokens: n_tokens + x.shape[1]].copy_(x_bins)


class HashLayer(torch.nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out, silu):
        super().__init__()

        self.proj = torch.nn.Parameter(
            torch.zeros((num_heads,dim_inp,dim_out), device='cuda'), 
            requires_grad=True)
        
        self.bias = torch.nn.Parameter(
            torch.zeros((1, 1, num_heads, dim_out), device='cuda'), 
            requires_grad=True)

        self.norm = torch.nn.LayerNorm(dim_out, device='cuda')

        std = (2.0 / dim_out) ** 0.5
        torch.nn.init.normal_(self.proj, mean=0.0, std=std)
        torch.nn.init.zeros_(self.bias)

        self.silu = torch.nn.SiLU() if silu else torch.nn.Identity()
        self.enable_residual = dim_inp == dim_out

    def forward(self, x):
        out = self.silu(torch.einsum('bnhd,hde->bnhe', x, self.proj) + self.bias)
        if self.enable_residual:
            out = out + x
        return self.norm(out)


class ThreshHead(torch.nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        self.reduce_proj = torch.nn.Parameter(
            torch.zeros((num_heads, dim), device='cuda'), 
            requires_grad=True)

    def forward(self, x):
        output = torch.einsum("bnhd,hd->bnh", x, self.reduce_proj)
        return torch.sigmoid(output)


class HashModule(torch.nn.Module):
    def __init__(self, num_heads, dims):
        super().__init__()
        n_layers = len(dims) - 1

        self.mlp = torch.nn.Sequential(*[
            HashLayer(num_heads, dims[i], dims[i+1], i < n_layers - 1)
            for i in range(n_layers)])

    def forward(self, x):
        embed = self.mlp(x)
        return embed


class Cache:
    def __init__(self, 
            batch_size, 
            max_tokens, 
            num_heads, 
            dims, 
            lru_budget=0.1,
            top_budget=0.025,
            dtype=torch.float, 
            device='cuda'):
        
        self.B = batch_size
        self.MaxT = max_tokens
        self.KH = num_heads
        self.dims = dims
        self.device = device
        
        self.query_hash = HashModule(num_heads, dims)
        self.key_hash = HashModule(num_heads, dims)

        self.key_cache = torch.zeros((batch_size, max_tokens, num_heads, dims[0]), dtype=dtype, device=device)
        self.value_cache = torch.zeros((batch_size, max_tokens, num_heads, dims[0]), dtype=dtype, device=device)
        
        num_uint32 = dims[-1]
        self.key_bins = torch.zeros((batch_size, max_tokens, num_heads, num_uint32), dtype=torch.int32, device=device)
        
        self.num_tokens = 0
        self.lru_size = int(lru_budget * max_tokens)
        self.lru_budget = self.lru_size - 1
        self.top_budget = int(top_budget * max_tokens)
        
        self.key_lru = torch.zeros((batch_size, self.lru_size, num_heads, dims[0]), dtype=dtype, device=device)
        self.value_lru = torch.zeros((batch_size, self.lru_size, num_heads, dims[0]), dtype=dtype, device=device)
        
        self.lru_ptr = torch.zeros((batch_size, num_heads), dtype=torch.int32, device=device)
        self.lru_indices = torch.full((batch_size, num_heads, self.lru_size), -1, dtype=torch.int32, device=device)

        # Async stuff
        self.pool = ThreadPoolExecutor(1)
        self.async_retrieve_future = None
        self.async_retrieve_stream = Stream()
        self.async_copy_stream = Stream()


    @torch.inference_mode()
    def async_retrieve_worker(self, query):
        with torch.cuda.stream(self.async_retrieve_stream):
            q_emb = self.query_hash(query)
            q_bin = packbits.packbits_kernel(q_emb)
            mask = attn_k8_q32.attn_k8_q32(q_bin, key_bins)

            topk_indices = torch.topk(mask, k=self.top_budget, dtype=torch.int32).indices

            lru_cache_kernel.update(
                topk_indices,
                self.lru_indices,
                self.lru_ptr,
                self.key_cache,
                self.value_cache,
                self.key_lru,
                self.value_lru,
                self.lru_budget,
                self.top_budget)


    def wait_async_retrieve(self):
        if self.async_retrieve_future is not None:
            self.async_retrieve_future.result()
            self.async_retrieve_future = None


    def retrieve(self, query):
        self.async_retrieve_future = self.pool.submit(self.async_retrieve_worker, query)


    @torch.inference_mode()
    def update(self, keys, values):
        B, T, KH, D = keys.shape
        start = self.num_tokens
        end = start + T
        
        with torch.cuda.stream(self.async_copy_stream):
            self.key_cache[:, start:end].copy_(keys, non_blocking=True)
            self.value_cache[:, start:end].copy_(values, non_blocking=True)

            if T == 1:
                self.key_lru[:, -1].copy_(keys, non_blocking=True)
                self.value_lru[:, -1].copy_(values, non_blocking=True)

        self.num_tokens += T
        self.wait_async_retrieve()

        return self.key_lru, self.value_lru
