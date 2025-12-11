import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 (保持不变)
labels = ['Qwen3\n1.7B', 'Qwen3\n4B', 'Qwen3\n8B', 'Qwen3\n14B', 'Qwen3\n4B-Think', 'GPT-OSS\n20B']
lru_budget_k = [0.65, 0.68, 0.67, 0.69, 0.66, 0.66]
lru_budget_2k = [0.87, 0.87, 0.87, 0.87, 0.90, 0.89]

# 2. 设置学术风格 (字体增大，移除默认加粗倾向)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2       # 稍微增加坐标轴线宽以匹配大字体
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['font.size'] = 15             

fig, ax = plt.subplots(figsize=(7, 5), dpi=640)

x = np.arange(len(labels))
width = 0.32

# 4. 绘制柱状图 (保持不变：斜线 和 其他图案)
rects1 = ax.bar(x - width/2, lru_budget_k, width, 
                label='LRU Budget=K', color='#E57A5D', edgecolor='black', linewidth=0.8,
                hatch='///') # 斜线背景

rects2 = ax.bar(x + width/2, lru_budget_2k, width, 
                label='LRU Budget=2K', color='#4B6483', edgecolor='black', linewidth=0.8,
                hatch='xx')  # 其他图案(交叉)

# 5. 添加数值标签 (Auto-labeling)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 稍微增加垂直偏移量适应大字体
                    textcoords="offset points",
                    ha='center', va='bottom',
                    # 【修改点2：字体变大，且改为正常粗细 (normal)】
                    fontsize=13, fontweight='normal') 

autolabel(rects1)
autolabel(rects2)

# 6. 调整坐标轴和标题
# 【修改点3：标题字体更大，且不加粗】
ax.set_title('LRU Eviction Hit Rate under Oracle Top-K (K=5%)', 
             fontsize=18, fontweight='normal', pad=20)

ax.set_xticks(x)
# 【修改点4：X轴标签字体增大】
ax.set_xticklabels(labels, fontsize=14) 
ax.set_ylim(0.2, 1.08) # 稍微增加Y轴上限，防止大号数字顶格

# Y轴网格线
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.4)
ax.set_axisbelow(True)

# 7. 图例设置
ax.legend(loc='lower center', 
          bbox_to_anchor=(0.5, 0.12),
          ncol=1, frameon=True, edgecolor='black', fancybox=False,
          # 【修改点5：图例字体增大】
          fontsize=14) 

plt.tight_layout()

# 显示或保存
plt.savefig('analysis/hit-rate-all-models.pdf', format='pdf', bbox_inches='tight')