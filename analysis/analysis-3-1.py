import os
import json
import torch
from collections import OrderedDict

DATA_COUNT = 10
LAYER_COUNT = 36
SUB_DIR = 'qwen3-14b'

# === 参数分离 ===
CACHE_RATIO = 0.1
EVAL_TOPK_RATIO = 0.05

for data_idx in range(DATA_COUNT):
    root_dir = f'/mnt/petrelfs/liwenhao/spotlight/attns/{SUB_DIR}/data-{data_idx}'

    # 记录三种策略的平均命中率
    avg_hits_fifo = [] 
    avg_hits_lru = []  
    avg_hits_lfu = []

    for layer_idx in range(LAYER_COUNT):
        path = os.path.join(root_dir, f'layer-{layer_idx}.pth')
        t_list = torch.load(path, map_location='cpu')
        
        max_seq_len = t_list[-1].numel()
        
        # === 状态初始化 ===
        lru_cache = OrderedDict()
        access_count = torch.zeros(max_seq_len, dtype=torch.long, device='cpu')
        
        layer_rates_fifo = []
        layer_rates_lru = []
        layer_rates_lfu = []

        # 遍历每一步
        for i in range(len(t_list) - 1):
            curr_scores = t_list[i]
            next_scores = t_list[i+1]
            current_len = curr_scores.numel()
            
            # Cache Budget
            cache_budget = int(current_len * CACHE_RATIO)
            if cache_budget < 1: cache_budget = 1
            
            # 定义“访问”
            active_indices = torch.topk(curr_scores, k=cache_budget).indices
            active_indices_list = active_indices.tolist()
            
            # === 1. Update LRU ===
            for idx in active_indices_list:
                if idx in lru_cache:
                    lru_cache.move_to_end(idx)
                else:
                    lru_cache[idx] = True
            while len(lru_cache) > cache_budget:
                lru_cache.popitem(last=False)

            # === 2. Update LFU ===
            access_count[active_indices] += 1
            access_count[current_len - 1] += 1

            # === Ground Truth ===
            eval_k = int(current_len * EVAL_TOPK_RATIO)
            if eval_k < 1: eval_k = 1
            truth_indices = set(torch.topk(next_scores[:current_len], k=eval_k).indices.tolist())

            if len(truth_indices) == 0: continue

            # === Predict & Eval ===
            # FIFO
            start_idx = max(0, current_len - cache_budget)
            pred_fifo = set(range(start_idx, current_len))
            
            # LRU
            pred_lru = set(lru_cache.keys())
            
            # LFU
            valid_counts = access_count[:current_len]
            pred_lfu = set(torch.topk(valid_counts, k=cache_budget).indices.tolist())

            layer_rates_fifo.append(len(pred_fifo & truth_indices) / len(truth_indices))
            layer_rates_lru.append(len(pred_lru & truth_indices) / len(truth_indices))
            layer_rates_lfu.append(len(pred_lfu & truth_indices) / len(truth_indices))

        # 汇总该层
        if len(layer_rates_lru) > 0:
            avg_hits_fifo.append(sum(layer_rates_fifo) / len(layer_rates_fifo))
            avg_hits_lru.append(sum(layer_rates_lru) / len(layer_rates_lru))
            avg_hits_lfu.append(sum(layer_rates_lfu) / len(layer_rates_lfu))
        else:
            # 防止空数据导致索引对不齐
            avg_hits_fifo.append(0.0)
            avg_hits_lru.append(0.0)
            avg_hits_lfu.append(0.0)

    # === 构造目标格式 (针对 LRU) ===
    # 如果需要保存 FIFO 或 LFU，只需替换下方的 avg_hits_lru 变量即可
    output_data = {}
    
    # 1. 填充每层数据
    for i, score in enumerate(avg_hits_lru):
        output_data[f"layer-{i}"] = score
    
    # 2. 计算并填充总平均值
    overall_avg = sum(avg_hits_lru) / len(avg_hits_lru) if avg_hits_lru else 0
    output_data["avg"] = overall_avg

    # 保存结果
    path = os.path.join(root_dir, "lru_cache.json")
    with open(path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Data {data_idx} Saved. Avg LRU: {overall_avg:.4f}")