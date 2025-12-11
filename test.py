import torch
from profiler import WallTime
from flash_attn import flash_attn_func


wt = WallTime(f"context-", cuda=0)

for prompt in [1024 * 2 ** i for i in range(12)]:

    q = torch.randn([1, 1, 28, 128], dtype=torch.bfloat16, device='cuda')
    
    proj = torch.nn.Sequential(
        torch.nn.Linear(128,128, device='cuda', dtype=torch.bfloat16),
        torch.nn.GELU(),
        torch.nn.Linear(128,128, device='cuda', dtype=torch.bfloat16)
    )


    for _ in range(3):
        with wt:
            q_bits = proj(q) > 0
    
    wt.result(postfix=prompt, detail=True)
    wt.reset()
  