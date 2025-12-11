import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 24,           # 设置基础字体大小 (默认 10)
    'axes.titlesize': 24,      # 设置坐标轴标题大小 (默认 12)
    'axes.labelsize': 24,      # 设置 x, y 轴标签大小 (默认 10)
    'xtick.labelsize': 24,     # 设置 x 轴刻度标签大小 (默认 10)
    'ytick.labelsize': 24,     # 设置 y 轴刻度标签大小 (默认 10)
    'legend.fontsize': 24      # 设置图例字体大小 (默认 10)
})

DATA_COUNT = 7
LAYER_COUNT = 24
SUB_DIR = 'qwen3-14b'

rate1 = []
rate2 = []

def get_data_by_layer(x, i):
    return [xx[f'layer-{i}'] for xx in x]

def compute_avg_std(x, i):
    data = [xx[f'layer-{i}'] for xx in x]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


for i in range(DATA_COUNT):
    root_dir = f'attns/{SUB_DIR}/data-{i}'
    rate_before = os.path.join(root_dir, 'summary.json')
    rate_after = os.path.join(root_dir, 'lru_cache.json')

    with open(rate_before, 'r') as f:
        rate_before = json.load(f)
        rate1.append(rate_before)

    with open(rate_after, 'r') as f:
        rate_after = json.load(f)
        rate2.append(rate_after)

rate1_means = []
rate1_stds = []
rate2_means = []
rate2_stds = []

layer_indices = list(range(LAYER_COUNT))

for layer_idx in layer_indices:
    mean1, std1 = compute_avg_std(rate1, layer_idx)
    rate1_means.append(mean1)
    rate1_stds.append(std1)
    
    mean2, std2 = compute_avg_std(rate2, layer_idx)
    rate2_means.append(mean2)
    rate2_stds.append(std2)

# --- 转换为Numpy数组 ---
rate1_means = np.array(rate1_means)
rate1_stds = np.array(rate1_stds)
rate2_means = np.array(rate2_means)
rate2_stds = np.array(rate2_stds)

plt.figure(figsize=(8, 5))

for layer_idx in layer_indices:
    rate1_data = get_data_by_layer(rate1, layer_idx)
    rate2_data = get_data_by_layer(rate2, layer_idx)

    # --- 移除散点图的标签，避免图例重复 ---
    plt.scatter(np.full((len(rate1_data),), layer_idx), rate1_data, 
                color='#996666', alpha=0.4, s=10, 
                label='_nolegend_') # 使用 '_nolegend_' 避免在图例中显示
                
    plt.scatter(np.full((len(rate2_data),), layer_idx), rate2_data, 
                color='#003366', alpha=0.4, s=10, 
                label='_nolegend_') # 使用 '_nolegend_' 避免在图例中显示

# --- 使用 plt.fill_between 替换 plt.errorbar ---

# 绘制 rate1 (w/o Index Shift) 的均值线和±1标准差区域
plt.plot(layer_indices, rate1_means, color='#996666', linewidth=1.2)
plt.fill_between(layer_indices, rate1_means - rate1_stds, rate1_means + rate1_stds,
                 color='#996666', alpha=0.2, 
                 label='5% LRU Cache Budget')
             
# 绘制 rate2 (w/ Index Shift) 的均值线和±1标准差区域
plt.plot(layer_indices, rate2_means, color='#003366', linewidth=1.2)
plt.fill_between(layer_indices, rate2_means - rate2_stds, rate2_means + rate2_stds,
                 color='#003366', alpha=0.2,
                 label='10% LRU Cache Budget')

plt.xlabel('Layer Index')
plt.ylabel('Overlap Coefficient')
plt.title(f'Temporal Coherence of the Top-5% Set')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(np.arange(0, LAYER_COUNT, 2))
plt.xlim(-1, LAYER_COUNT)

plt.ylim(0.4, 1)


output_path = f'attns/{SUB_DIR}/fig1.png'

# 提高 `figsize` 和 `dpi` 以适应更大的字体
# 字体变大后，可能需要更大的画布来避免元素重叠
# 原始 (8, 5)，我们将其调整为 (12, 7.5) 以保持比例并提供更多空间
plt.gcf().set_size_inches(12, 7.5)

plt.savefig(output_path, dpi=640)

# 提示：运行后检查 fig1.png。
# 如果标签或标题仍然重叠，您可能需要：
# 1. 进一步增大 figsize (例如 (14, 9))
# 2. 在 savefig 之前调用 plt.tight_layout()
#    (但 tight_layout 有时会与自定义布局冲突，所以先尝试调整 figsize)