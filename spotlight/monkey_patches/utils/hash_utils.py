import torch


def compute_lsh_score(q_hash, k_hash, random_query_index):
    group_size = q_hash.shape[2] // k_hash.shape[2]

    if random_query_index is not None:
        q_hash = q_hash[:, random_query_index]

    q_hash = q_hash.transpose(1, 2)
    k_hash = k_hash.transpose(1, 2)
    k_hash = k_hash.unsqueeze(2).expand(-1, -1, group_size, -1, -1).flatten(1,2)
    
    q_bin = (q_hash > 0).bfloat16() * 2 - 1
    k_bin = (k_hash > 0).bfloat16() * 2 - 1
    
    q_soft = q_hash / (1 + q_hash.abs())
    k_soft = k_hash / (1 + k_hash.abs())
    
    q_final = (q_bin - q_soft).detach() + q_soft
    k_final = (k_bin - k_soft).detach() + k_soft

    scale = 1 / q_hash.shape[-1] ** 0.5
    sim = q_final @ k_final.transpose(-1, -2) * scale
    
    return sim


def compute_attn_score(q, k, random_query_index):
    score = []
    num_heads = q.shape[2]
    group_size = num_heads // k.shape[2]

    if random_query_index is not None:
        q = q[:, random_query_index]

    for head_idx in range(num_heads):
        q_head = q[..., head_idx, :]
        k_head = k[..., head_idx // group_size , :]
        head_score = q_head @ k_head.transpose(-1,-2)
        score.append(head_score)

    return torch.stack(score, dim=1)


class HashLayer(torch.nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out, silu, dropout):
        super().__init__()

        self.proj = torch.nn.Parameter(
            torch.zeros((num_heads,dim_inp,dim_out), dtype=torch.float, device='cuda'), 
            requires_grad=True)
        
        self.bias = torch.nn.Parameter(
            torch.zeros((1, 1, num_heads, dim_out), 
            dtype=torch.float, 
            device='cuda'), 
            requires_grad=True)

        std = (2.0 / dim_out) ** 0.5
        torch.nn.init.normal_(self.proj, mean=0.0, std=std)
        torch.nn.init.zeros_(self.bias)

        self.drop = torch.nn.Dropout(dropout)
        self.silu = torch.nn.SiLU() if silu else torch.nn.Identity()
        self.enable_residual = dim_inp == dim_out

    def forward(self, x):
        out = self.silu(self.drop(torch.einsum('bnhd,hde->bnhe', x, self.proj) + self.bias))
        if self.enable_residual:
            out = out + x
        return out


class HashModule(torch.nn.Module):
    def __init__(self, num_heads, dims, dropout):
        super().__init__()
        n_layers = len(dims) - 1
        self.mlp = torch.nn.Sequential(*[
            HashLayer(num_heads, dims[i], dims[i+1], i < n_layers - 1, dropout)
            for i in range(n_layers)])
            
    def forward(self, x):
        return self.mlp(x)