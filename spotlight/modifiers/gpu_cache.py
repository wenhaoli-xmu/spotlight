import torch

INIT_SEQ_LEN = 1024 * 2048 + 128
    

class KVHash:

    init_seq_length = INIT_SEQ_LEN

    def __init__(
            self, 
            batch_size, 
            num_kv_heads, 
            head_dim, 
            num_layers, 
            dtype=torch.bfloat16,
            device=None,
        ):
        
        self.num_kv_heads = num_kv_heads
        self.length = [0 for _ in range(num_layers)]
        self.max_length = self.init_seq_length
        self.num_layers = num_layers

        if device == None:
            device = ['cuda' for _ in range(num_layers)]

        self.key_cache = [
            torch.zeros((
                batch_size,
                self.init_seq_length, 
                num_kv_heads, 
                head_dim), 
            dtype=dtype, 
            device=dev)
            for dev in device]

    def update(self, layer_idx, key):
        start = self.length[layer_idx]
        end = start + key.shape[1]

        self.key_cache[layer_idx][:, start: end] = key
        self.length[layer_idx] = end

        return self.key_cache[layer_idx][:, :end]
    

@torch.compile
def topk_and_gather(attn, keys, vals, budget):
    ind = attn.topk(k=budget, dim=1, sorted=False).indices
    ind = ind.unsqueeze(-1).expand(-1,-1,-1,128)
    key_topk = torch.gather(keys, dim=1, index=ind)
    val_topk = torch.gather(vals, dim=1, index=ind)
    return key_topk, val_topk


class KVCache:
    init_seq_length = INIT_SEQ_LEN

    def __init__(
            self, 
            batch_size, 
            num_kv_heads, 
            head_dim, 
            num_layers,
            dtype=torch.bfloat16, 
            device=None,
            **kwargs,
        ):
        
        self.num_kv_heads = num_kv_heads
        self.length = [0 for _ in range(num_layers)]
        self.max_length = self.init_seq_length
        self.num_layers = num_layers

        if device == None:
            device = ['cuda' for _ in range(num_layers)]
        self.device = device

        self.key_cache = [
            torch.zeros((
                batch_size,
                self.init_seq_length, 
                num_kv_heads, 
                head_dim), 
            dtype=dtype, 
            device=dev)
            for dev in device]
        
        self.val_cache = [
            torch.zeros((
                batch_size, 
                self.init_seq_length, 
                num_kv_heads, 
                head_dim), 
            dtype=dtype, 
            device=dev)
            for dev in device]
        
    def gather(self, layer_idx, attn, budget):
        keys_topk, vals_topk = topk_and_gather(
            attn, 
            self.key_cache[layer_idx], 
            self.val_cache[layer_idx], 
            budget)
        return keys_topk, vals_topk
        
    def update(self, layer_idx, key, val):
        start = self.length[layer_idx]
        end = start + key.shape[1]

        self.key_cache[layer_idx][:, start: end, ...] = key
        self.val_cache[layer_idx][:, start: end, ...] = val
        self.length[layer_idx] = end

        return (
            self.key_cache[layer_idx][:, :end], 
            self.val_cache[layer_idx][:, :end])
        

