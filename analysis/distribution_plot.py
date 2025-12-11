import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. 设置绘图参数
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.5 
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(6,4), dpi=300)

# 创建模拟数据
x = np.linspace(0, 100, 1000)
y = norm.pdf(x, 50, 15) * 100 

# 定义浅色系颜色 (Light Palette)
color_low = '#A6CEE3'   # 浅蓝
color_mid = '#B2DF8A'   # 浅绿
color_high = '#FB9A99'  # 浅红/粉
text_color_low = '#1F78B4'  # 深蓝 (用于文字)
text_color_mid = '#33A02C'  # 深绿 (用于文字)
text_color_high = '#E31A1C' # 深红 (用于文字)

# 定义区域边界
low_cutoff = 30
high_cutoff = 70

# --- 区域 1: Mean Pooling ---
ax.fill_between(x, 0, y, where=(x <= low_cutoff), 
                color=color_low, alpha=0.9, label='Mean Pooling Focus', linewidth=0)
ax.text(15, 0.5, "Mean Pooling\nLow Freq\nBlurring Effect", 
        ha='center', va='center', color=text_color_low, fontweight='bold', fontsize=12)

# --- 区域 2: Effective Information ---
ax.fill_between(x, 0, y, where=((x > low_cutoff) & (x < high_cutoff)), 
                color=color_mid, alpha=0.9, label='Effective Information', linewidth=0)
ax.text(50, 1.0, "Effective Information\nSemantic\nVertical Pattern Lives Here", 
        ha='center', va='center', color=text_color_mid, fontweight='bold', fontsize=12)

# --- 区域 3: Max Pooling ---
ax.fill_between(x, 0, y, where=(x >= high_cutoff), 
                color=color_high, alpha=0.9, label='Max Pooling Focus', linewidth=0)
ax.text(85, 0.5, "Max Pooling\nHigh Freq\nNoise Sensitivity", 
        ha='center', va='center', color=text_color_high, fontweight='bold', fontsize=12)

# 美化正态分布曲线：使用极细的灰色线条代替粗黑线
ax.plot(x, y, color='#777777', linewidth=0.8, alpha=0.6)

# 添加分隔线 (使用白色使其看起来更自然)
ax.vlines(x=low_cutoff, ymin=0, ymax=norm.pdf(low_cutoff, 50, 15)*100, colors='white', linestyles='-', linewidth=1.5, alpha=0.8)
ax.vlines(x=high_cutoff, ymin=0, ymax=norm.pdf(high_cutoff, 50, 15)*100, colors='white', linestyles='-', linewidth=1.5, alpha=0.8)

# 装饰图表
ax.set_xlabel("Key Feature Frequency", fontsize=13, color='#555555')
ax.set_ylabel("Information Density", fontsize=13, color='#555555')
ax.set_yticks([]) 
ax.set_xlim(0, 100)
ax.set_ylim(0, 3)

# 去除多余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#AAAAAA')

plt.tight_layout()

# 保存
plt.savefig('analysis/distribution_plot.pdf', format='pdf')