import matplotlib.pyplot as plt
import numpy as np

# === 1. 风格配置 ===
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 15  # 保持全局 15

# === 2. 定义配色 ===
COLOR_DENSE = '#4B6483'
COLOR_MOE = '#E57A5D'

# === 3. 数据准备 ===
models = [
    {"name": "Llama-2-70B", "year": 2023.55, "ratio": 27.5, "type": "Dense"},
    {"name": "Llama-3-70B", "year": 2024.30, "ratio": 27.5, "type": "Dense"},
    {"name": "Mixtral 8x7B", "year": 2024.05, "ratio": 60.0, "type": "MoE"},
    {"name": "DeepSeek-V2/V3", "year": 2024.85, "ratio": 80.5, "type": "MoE"},
]

dense_x = [m['year'] for m in models if m['type'] == 'Dense']
dense_y = [m['ratio'] for m in models if m['type'] == 'Dense']
moe_x = [m['year'] for m in models if m['type'] == 'MoE']
moe_y = [m['ratio'] for m in models if m['type'] == 'MoE']

# === 4. 绘图 ===
fig, ax = plt.subplots(figsize=(7, 5), dpi=640)

# --- 背景区间 ---
ax.axhspan(20, 35, color=COLOR_DENSE, alpha=0.1, lw=0)
ax.axhspan(55, 85, color=COLOR_MOE, alpha=0.1, lw=0)

# --- 背景文字说明 ---
# 【对齐修改：字号从 15 降为 14 (对齐图例大小)】
ax.text(2024.95, 22, "Dense Paradigm\n(Attention is minor)", 
        color=COLOR_DENSE, fontsize=14, va='center', ha='right', 
        fontweight='normal', alpha=0.9)

ax.text(2023.45, 78, "MoE Paradigm\n(Attention dominates)", 
        color=COLOR_MOE, fontsize=14, va='center', ha='left', 
        fontweight='normal', alpha=0.9)

# --- 趋势线 ---
ax.plot(dense_x, dense_y, color=COLOR_DENSE, linestyle='--', linewidth=2.5, alpha=0.8)
ax.plot(moe_x, moe_y, color=COLOR_MOE, linestyle='--', linewidth=2.5, alpha=0.8)

# --- 箭头 ---
ax.annotate('', xy=(2024.9, 82), xytext=(2024.85, 80.5),
            arrowprops=dict(arrowstyle='->', color=COLOR_MOE, lw=2.5))

# --- 散点 ---
ax.scatter(dense_x, dense_y, s=220, color=COLOR_DENSE, zorder=10, edgecolors='white', linewidth=1.5)
ax.scatter(moe_x, moe_y, s=220, color=COLOR_MOE, zorder=10, edgecolors='white', linewidth=1.5)

# --- 数据标签 ---
for m in models:
    label_color = 'black' 
    
    if m['type'] == 'Dense':
        ax.annotate(m['name'], (m['year'], m['ratio']), 
                    xytext=(0, -22), textcoords='offset points',
                    ha='center', va='top',
                    # 【对齐修改：字号从 15 降为 13 (对齐上一张图的数值标签)】
                    fontsize=13, fontweight='normal', color=label_color)
    else:
        offset_x = -15 if "DeepSeek" in m['name'] else 0
        ax.annotate(m['name'], (m['year'], m['ratio']), 
                    xytext=(offset_x, 18), textcoords='offset points',
                    ha='center', va='bottom',
                    # 【对齐修改：字号从 15 降为 13 (对齐上一张图的数值标签)】
                    fontsize=13, fontweight='normal', color=label_color)

# === 5. 装饰轴 ===
ax.set_xlim(2023.4, 2025.0) 
ax.set_ylim(0, 100)

ax.set_xticks([2023.5, 2024.0, 2024.5, 2025.0])
# 【对齐修改：字号从 15 降为 14 (对齐上一张图的轴标签)】
ax.set_xticklabels(['2023 H2', '2024 H1', '2024 H2', '2025'], fontsize=14)

# 【对齐修改：Y轴标题 17 -> 15 (基础字号), 主标题 20 -> 18 (对齐上一张图标题)】
ax.set_ylabel(r"Attention FLOPs / Total Active FLOPs (%)", fontsize=15, fontweight='normal')
ax.set_title("The Rising Cost of Attention in MoE Era", fontsize=18, pad=20, fontweight='normal')

plt.tight_layout()
plt.savefig('analysis/moe_trend_aligned.pdf', format='pdf', bbox_inches='tight')