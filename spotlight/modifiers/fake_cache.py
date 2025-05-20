import torch

INIT_SEQ_LEN = 1024 * 2048 + 128
BOUND_SEQ_LEN = 1024 * 2048 + 128
    

class KVHash:

    init_seq_length = INIT_SEQ_LEN
    bound_seq_length = BOUND_SEQ_LEN

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

        self.batch_size = batch_size

        self.key_cache = [
            torch.zeros((
                1, 
                self.init_seq_length, 
                num_kv_heads, 
                head_dim), 
            dtype=dtype, 
            device=dev)
            for dev in device]

    def update(self, layer_idx, key):
        start = self.length[layer_idx]
        end = start + key.shape[1]

        self.key_cache[layer_idx][:, start: end] = key[:1]
        self.length[layer_idx] = end

        return self.key_cache[layer_idx][:, :end].expand(self.batch_size, -1, -1, -1)


class KVCache:
    init_seq_length = INIT_SEQ_LEN
    bound_seq_length = BOUND_SEQ_LEN

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

        self.batch_size = batch_size

        self.key_cache = [
            torch.zeros((
                1, 
                self.init_seq_length, 
                num_kv_heads, 
                head_dim), 
            dtype=dtype, 
            device=dev)
            for dev in device]
        
        self.val_cache = [
            torch.zeros((
                1, 
                self.init_seq_length, 
                num_kv_heads, 
                head_dim), 
            dtype=dtype, 
            device=dev)
            for dev in device]

    def update(self, layer_idx, key, val):
        start = self.length[layer_idx]
        end = start + key.shape[1]

        self.key_cache[layer_idx][:, start: end, ...] = key[:1]
        self.val_cache[layer_idx][:, start: end, ...] = val[:1]
        self.length[layer_idx] = end

        return (
            self.key_cache[layer_idx][:, :end].expand(self.batch_size, -1, -1, -1), 
            self.val_cache[layer_idx][:, :end].expand(self.batch_size, -1, -1, -1))
        

