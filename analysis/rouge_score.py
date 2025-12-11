import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the output directory exists
os.makedirs('analysis', exist_ok=True)

# 1. Set plot parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 15

# 2. Data Preparation
# Model names aligned with the rows: 4B-Think, 4B, 8B, 14B
model_names = ["Qwen3-4B-\nThinking-2507", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]

# Configs aligned with columns: Origin, Top-5%, Top-5% (Blk 8), Top-5% (Blk 64)
configs = ["Origin", "Top-5%", "Top-5% (Block-8)", "Top-5% (Block-64)"]

# Data Extraction from Image (Rows correspond to models, Columns to configs)
# Table 1: Qasper (Rouge-L)
data_qasper_raw = [
    [11.70, 11.59, 10.58, 10.96], # 4B-Think
    [39.40, 40.86, 37.25, 38.58], # 4B
    [47.84, 47.25, 45.62, 46.74], # 8B
    [46.17, 46.65, 45.13, 45.67]  # 14B
]

# Table 2: QMSUM (Rouge-L)
data_qmsum_raw = [
    [19.18, 20.51, 18.62, 19.25], # 4B-Think
    [21.22, 22.68, 20.59, 20.89], # 4B
    [20.71, 20.56, 19.68, 19.82], # 8B
    [20.80, 20.14, 19.37, 19.61]  # 14B
]

# Table 3: Repobench (Rouge-L)
data_repo_raw = [
    [3.94,  6.00,  3.12,  3.75],  # 4B-Think
    [3.01,  4.60,  0.00,  2.85],  # 4B
    [1.60,  2.00,  0.00,  1.20],  # 8B
    [8.80,  10.40, 5.40,  6.80]   # 14B
]

def process_data(data_raw):
    # Transpose data to fit the plotting logic: list of configs, each containing list of model scores
    return np.array(data_raw).T

data_qasper = process_data(data_qasper_raw)
data_qmsum = process_data(data_qmsum_raw)
data_repo = process_data(data_repo_raw)

dataset_list = [
    {"name": "Qasper (Rouge-L)", "data": data_qasper, "file_suffix": "Qasper"},
    {"name": "QMSum (Rouge-L)", "data": data_qmsum, "file_suffix": "QMSUM"},
    {"name": "RepoBench (Rouge-L)", "data": data_repo, "file_suffix": "Repobench"}
]

colors = ['#0B1F41', '#B85B4B', '#E8C99B', '#7A6071']
hatches = ['/', '\\', 'x', 'o']

# 3. Plotting and Saving separate PDFs
bar_width = 0.2
x = np.arange(len(model_names))

output_files = []

for dataset in dataset_list:
    # Create a separate figure for each dataset
    fig, ax = plt.subplots(figsize=(6, 6))
    data = dataset["data"]
    name = dataset["name"]
    suffix = dataset["file_suffix"]

    for i in range(len(configs)):
        offset = (i - len(configs)/2) * bar_width + bar_width/2
        ax.bar(x + offset, data[i], bar_width, label=configs[i], 
               color=colors[i], hatch=hatches[i], edgecolor='black', linewidth=0.5)

    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    
    # Updated Y-label for Rouge-L tasks
    ax.set_ylabel("Rouge-L Score")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # Removed scientific notation since Rouge-L is typically 0-100
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False, fontsize=12)
    
    plt.tight_layout()
    
    # Construct filename
    filename = f"analysis/rouge_score_{suffix}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    output_files.append(filename)
    plt.close(fig)

print(f"Saved files: {output_files}")