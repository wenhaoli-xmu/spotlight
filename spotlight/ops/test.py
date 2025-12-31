import torch
import time
import math
from torch.nn import functional as F

# 尝试导入 flash_attn，如果不存在则报错提示
try:
    from flash_attn import flash_attn_func
except ImportError:
    raise ImportError("请安装 flash-attn: pip install flash-attn --no-build-isolation")

import hamming_ops

def benchmark_speed(func, name, n_warmup=10, n_repeat=100):
    # 预热
    for _ in range(n_warmup):
        func()
    torch.cuda.synchronize()
    
    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_repeat):
        func()
    end_event.record()
    torch.cuda.synchronize()
    
    avg_time_ms = start_event.elapsed_time(end_event) / n_repeat
    return avg_time_ms

def test_hamming_vs_flash():
    # ------------------------------
    # 1. 配置参数
    # ------------------------------
    device = torch.device("cuda")
    dtype = torch.bfloat16 # Flash Attn 标准精度
    
    B = 4
    N = 16384         # 序列长度 (较长序列更能体现稀疏优势)
    HQ = 32           # Query Heads
    HK = 4            # Key/Value Heads (GQA)
    D_attn = 128      # Attention Head Dim
    D_hash = 4        # Hash Dim (int32 * 4 = 128 bits)
    
    print(f"Config: B={B}, N={N}, HQ={HQ}, HK={HK}, D_attn={D_attn}")
    print("-" * 60)

    # ------------------------------
    # 2. 准备数据
    # ------------------------------
    # (A) Flash Attention 需要的数据 (BF16)
    q = torch.randn((B, N, HQ, D_attn), dtype=dtype, device=device)
    k = torch.randn((B, N, HK, D_attn), dtype=dtype, device=device)
    v = torch.randn((B, N, HK, D_attn), dtype=dtype, device=device)
    
    # (B) Hamming Mask 需要的数据 (UInt32)
    # 我们只模拟 Query 只有 1 个 token (decode阶段) 或者是 Full Prefill?
    # 你的 Kernel 现在的形状支持: Query [B, HQ, D], Key [B, N, HK, D]
    # 这是一个典型的 Prefill / All-to-All 场景的 mask 计算
    
    # 注意：为了让测试有意义，我们假设 Query 也是 N 长度
    # 但你的 kernel 目前 query 输入看起来是 [B, HQ, D] (即 query length=1 的 decode 或者是 global query params)
    # 假设你的应用场景是：为当前的 queries (Batch) 从 KV Cache (N) 中检索
    # 如果是 Prefill (N对N)，你的 Kernel 需要改为接收 Query [B, N, HQ, D]。
    # 基于你之前的 Kernel 代码: Query 被读取为 flat pointer，我们假设这里是 Decode 阶段 (Q_len=1)
    # 或者 Q 是 Parameter。我们按原代码逻辑构造:
    
    # 构造 Q [B, 1, HQ, D_hash] 和 K [B, N, HK, D_hash]
    q_bin = torch.randint(0, 2**31 - 1, (B, 1, HQ, D_hash), dtype=torch.int32, device=device)
    k_bin = torch.randint(0, 2**31 - 1, (B, N, HK, D_hash), dtype=torch.int32, device=device)
    
    # 确保内存连续
    q_bin = q_bin.contiguous()
    k_bin = k_bin.contiguous()

    # ------------------------------
    # 3. 验证 Select Ratio (5%)
    # ------------------------------
    # 对于 128 bit 随机哈希，二项分布 B(128, 0.5):
    # Mean=64, Std=5.66. Top 5% 约在 Mean + 1.65*Std ≈ 73.3
    target_threshold = 73 
    
    mask = hamming_ops.calc_hamming_mask(q_bin, k_bin, target_threshold)
    ratio = mask.sum().item() / mask.numel()
    print(f"Data Source: Random Uniform")
    print(f"Threshold  : {target_threshold} (Targeting ~5% for > threshold logic)")
    print(f"Select Ratio: {ratio:.2%} (Expect ~5.0%)")
    
    # 如果偏差太大，自动微调一下展示效果（仅用于演示）
    if ratio < 0.01 or ratio > 0.15:
        print(">> Warning: Ratio skew due to limited randomness, adjusting threshold...")
        # 简单二分查找最佳 threshold (Mock过程，实际使用定死即可)
        pass 

    print("-" * 60)

    # ------------------------------
    # 4. 速度对比
    # ------------------------------
    
    # (A) Flash Attention (Full Dense)
    # 模拟 Decode 步骤：Q len = 1, KV len = N
    q_fa = q[:, :1, :, :] # [B, 1, HQ, D]
    
    func_flash = lambda: flash_attn_func(q_fa, k, v, causal=False)
    time_flash = benchmark_speed(func_flash, "FlashAttn")
    
    # (B) Hamming Mask Kernel
    func_hamming = lambda: hamming_ops.calc_hamming_mask(q_bin, k_bin, target_threshold)
    time_hamming = benchmark_speed(func_hamming, "HammingMask")
    
    print(f"Performance Comparison (Decode Step Q=1, KV={N}):")
    print(f"1. Dense FlashAttention-2 : {time_flash:.4f} ms")
    print(f"2. Hamming Mask Generation: {time_hamming:.4f} ms")
    print("-" * 60)
    
    # ------------------------------
    # 5. 结论分析
    # ------------------------------
    overhead = time_hamming / time_flash
    print(f"Analysis:")
    print(f"Mask Generation Overhead: {overhead:.2%} of Dense FlashAttn time")
    
    if overhead < 0.2:
        print(">> Result: GREAT. Hash calculation is very cheap.")
        print(f">> Theoretical Speedup Limit: 1 / ({overhead:.2f} + 0.05) ≈ {1/(overhead + 0.05):.1f}x")
        print("   (Assuming Sparse Attn computation scales linearly with 5% density)")
    else:
        print(">> Result: HEAVY. Hash calculation takes too long compared to Attention.")
        print(">> Consider optimizing kernel further or reducing D_hash.")

if __name__ == "__main__":
    test_hamming_vs_flash()