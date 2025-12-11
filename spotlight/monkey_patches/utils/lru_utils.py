import torch


def compute_lsh(q_hash, k_hash, gamma=64):
    q_hash = q_hash * gamma / (1 + q_hash.abs())
    k_hash = k_hash * gamma / (1 + k_hash.abs())
    sim = q_hash @ k_hash.transpose(-1,-2)
    return sim


class LRUCache:
    def __init__(
            self, 
            budget: float, 
            max_length: int, 
            num_heads: int, 
            head_dim: int):
        

        shape_full_cache = (1, max_length, num_heads, head_dim)
        shape_lru_cache = (1, int(budget * max_length), num_heads, head_dim)
        
        self.keys = torch.zeros(shape_full_cache, dtype=torch.bfloat16, device='cuda')
        self.vals = torch.zeros(shape_full_cache, dtype=torch.bfloat16, device='cuda')
        self.lru_cache = torch.zeros(shape_lru_cache, dtype=torch.bfloat16, device='cuda')
        self.budget = budget


    def update(self, ques, keys, vals):
        if ques.shape[self.seq_dim] == 1:
            # decoding

        else:
            seq_len = ques.shape[1]
            assert keys.shape[1] == vals.shape[1]

            self.keys[:, :]
            return keys, vals
