import os
import tqdm
import torch
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# === 1. 风格配置 (严格对齐之前的图表) ===
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2       # 坐标轴线宽
plt.rcParams['xtick.major.width'] = 1.2    # X轴刻度线宽
plt.rcParams['ytick.major.width'] = 1.2    # Y轴刻度线宽
plt.rcParams['font.size'] = 15             # 全局基础字号

# 定义对齐的配色方案
COLOR_K = '#E57A5D'
COLOR_2K = '#4B6483'
COLOR_RAND = 'gray'

DATA_COUNT = 10
LAYER_COUNT = 36
TOPK_LIST = [0.01, 0.04, 0.08, 0.12, 0.16, 0.20]
SUB_DIR = 'qwen3-4b-thinking'
BASE_ROOT_DIR = '/mnt/petrelfs/liwenhao/spotlight/attns'

def process_layer(layer_idx, root_dir, topk_ratio):
    """
    使用真实的 LRU 模拟逻辑处理单个层的数据。
    """
    path = os.path.join(root_dir, f'layer-{layer_idx}.pth')

    try:
        t_list = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"加载失败 {path}: {e}")
        return 0.0, 0.0, 0.0

    if not t_list:
        return 0.0, 0.0, 0.0

    total_elements = t_list[0].numel()
    if total_elements == 0:
        return 0.0, 0.0, 0.0

    # === 定义预算 ===
    budget_k = int(total_elements * topk_ratio)
    budget_2k = int(total_elements * topk_ratio * 3)
    
    budget_k = max(1, budget_k)
    budget_2k = max(1, budget_2k)

    cache_k = OrderedDict()
    cache_2k = OrderedDict()

    hits_k = []
    hits_2k = []

    rate_random = float(budget_k) / total_elements

    for i in range(len(t_list) - 1):
        curr_scores = t_list[i]
        next_scores = t_list[i+1]

        indices_k = torch.topk(curr_scores, k=budget_k).indices.tolist()
        indices_2k = torch.topk(curr_scores, k=budget_2k).indices.tolist()

        for idx in indices_k:
            if idx in cache_k:
                cache_k.move_to_end(idx)
            else:
                cache_k[idx] = True
        while len(cache_k) > budget_k:
            cache_k.popitem(last=False)

        for idx in indices_2k:
            if idx in cache_2k:
                cache_2k.move_to_end(idx)
            else:
                cache_2k[idx] = True
        while len(cache_2k) > budget_2k:
            cache_2k.popitem(last=False)

        truth_indices = set(torch.topk(next_scores, k=budget_k).indices.tolist())
        
        if not truth_indices:
            continue

        pred_k = set(cache_k.keys())
        pred_2k = set(cache_2k.keys())

        hits_k.append(len(pred_k & truth_indices) / len(truth_indices))
        hits_2k.append(len(pred_2k & truth_indices) / len(truth_indices))

    avg_k = sum(hits_k) / len(hits_k) if hits_k else 0.0
    avg_2k = sum(hits_2k) / len(hits_2k) if hits_2k else 0.0

    return avg_k, avg_2k, rate_random


def main():
    rate_data_topk_nlayers = []     
    rate_data_topk_nlayers2 = []    
    rate_data_topk_nlayers_random = []

    # === 多进程处理 ===
    for data_idx in tqdm.tqdm(range(DATA_COUNT), desc="Processing data"):
        root_dir = f'{BASE_ROOT_DIR}/{SUB_DIR}/data-{data_idx}'
        
        rate_topk_nlayers = []
        rate_topk_nlayers2 = []
        rate_topk_nlayers_random = []

        for topk in tqdm.tqdm(TOPK_LIST, desc=f"Data {data_idx} TopK", leave=False):
            tasks = [(layer_idx, root_dir, topk) for layer_idx in range(LAYER_COUNT)]

            with mp.Pool() as pool:
                results = pool.starmap(process_layer, tasks)
            
            rate_avgs = [res[0] for res in results]         
            rate_avgs2 = [res[1] for res in results]        
            rate_avgs_random = [res[2] for res in results]  

            rate_topk_nlayers.append(rate_avgs)
            rate_topk_nlayers2.append(rate_avgs2)
            rate_topk_nlayers_random.append(rate_avgs_random)

        rate_data_topk_nlayers.append(rate_topk_nlayers)
        rate_data_topk_nlayers2.append(rate_topk_nlayers2)
        rate_data_topk_nlayers_random.append(rate_topk_nlayers_random)

    # === 数据聚合 ===
    arr_k = np.array(rate_data_topk_nlayers)
    arr_2k = np.array(rate_data_topk_nlayers2)
    arr_rand = np.array(rate_data_topk_nlayers_random)

    mean_per_data_k = arr_k.mean(axis=-1)
    mean_per_data_2k = arr_2k.mean(axis=-1)
    mean_per_data_rand = arr_rand.mean(axis=-1)

    global_mean_k = mean_per_data_k.mean(axis=0)
    global_std_k = mean_per_data_k.std(axis=0)

    global_mean_2k = mean_per_data_2k.mean(axis=0)
    global_std_2k = mean_per_data_2k.std(axis=0)

    global_mean_rand = mean_per_data_rand.mean(axis=0)
    global_std_rand = mean_per_data_rand.std(axis=0)

    # === 绘图部分 (Polished & Aligned) ===
    plt.figure(figsize=(6, 5), dpi=640)

    # 1. 散点 (可视化分布) - 使用新配色
    for i, topk in enumerate(TOPK_LIST):
        plt.scatter(np.full(mean_per_data_k.shape[0], topk), mean_per_data_k[:, i],
                    color=COLOR_K, alpha=0.25, s=20, edgecolors='none')
    for i, topk in enumerate(TOPK_LIST):
        plt.scatter(np.full(mean_per_data_2k.shape[0], topk), mean_per_data_2k[:, i],
                    color=COLOR_2K, alpha=0.25, s=20, edgecolors='none')

    # 2. Errorbar 曲线
    # Random Baseline
    plt.errorbar(TOPK_LIST, global_mean_rand, yerr=global_std_rand, fmt='--', color=COLOR_RAND,
                 linewidth=2, capsize=4, alpha=0.8,
                 label=r'Random Baseline')

    # LRU (Budget = K) -> Orange (#E57A5D)
    plt.errorbar(TOPK_LIST, global_mean_k, yerr=global_std_k, fmt='-o', color=COLOR_K,
                 linewidth=2.5, markersize=7, capsize=4, markeredgecolor='white', markeredgewidth=1,
                 label=r'LRU Cache (Budget $k$)')

    # LRU (Budget = 2K) -> Blue (#4B6483)
    plt.errorbar(TOPK_LIST, global_mean_2k, yerr=global_std_2k, fmt='-s', color=COLOR_2K,
                 linewidth=2.5, markersize=7, capsize=4, markeredgecolor='white', markeredgewidth=1,
                 label=r'LRU Cache (Budget $2k$)')

    # 3. 标签与标题 (对齐字号)
    # X/Y Label: size 15, normal weight
    plt.xlabel('Top-K Ratio', fontsize=15, fontweight='normal')
    plt.ylabel('Hit Rate', fontsize=15, fontweight='normal')
    
    # Title: size 18, normal weight
    plt.title('LRU Eviction Hit Rate with Oracle Top-K', fontsize=18, fontweight='normal', pad=20)

    # 4. 图例 (对齐字号)
    # Legend: size 14
    plt.legend(loc='lower right', fontsize=14, framealpha=0.95, edgecolor='black', fancybox=False)

    # 5. 网格与刻度
    plt.grid(True, which='major', linestyle='--', alpha=0.4, color='grey')
    
    # Ticks: size 14
    plt.xticks(TOPK_LIST, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlim(0, TOPK_LIST[-1] + 0.02)
    plt.ylim(0.0, 1.05)

    plt.tight_layout()

    # 保存
    output_path = f'analysis/hit_rate_all_ratio.pdf'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"图像已保存至: {output_path}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        print(f"注意: 设置 'spawn' 失败 (可能已设置): {e}")

    main()