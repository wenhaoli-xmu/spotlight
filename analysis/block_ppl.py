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
model_names = ["Qwen3-4B-\nThinking-2507", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
configs = ["Full Cache", "Top-5%", "Top-5% (Block-8)", "Top-5% (Block-64)"]

data_math_raw = [
    ["167970", "142616.56138955228", "160795", "165848"],
    ["99245", "85794", "95158.83790144105", "97534.23422327096"],
    ["48106", "43115", "46666", "47867.162968185"],
    ["30870", "26256", "29729.20934439762", "30367"]
]
data_code_raw = [
    ["257164", "225155", "247560.63374370776", "253943.1822874882"],
    ["636478", "537570", "612104", "630330"],
    ["397292", "326411", "379679", "392524"],
    ["151710", "130148", "145732", "150092"]
]
data_book_raw = [
    ["1637187", "1324107", "1535175", "1598034"],
    ["2809641", "2124733.256244797", "2598165.805258724", "2741598"],
    ["1131900", "927020", "1067433", "1110376"],
    ["603446", "492337", "574546", "594113"]
]

def clean_and_convert(data_raw):
    data_clean = []
    for row in data_raw:
        clean_row = []
        for item in row:
            clean_val = float(str(item).split()[0])
            clean_row.append(clean_val)
        data_clean.append(clean_row)
    return np.array(data_clean).T

data_math = clean_and_convert(data_math_raw)
data_code = clean_and_convert(data_code_raw)
data_book = clean_and_convert(data_book_raw)

dataset_list = [
    {"name": "Perplexity on Math (Proof-Pile)", "data": data_math, "file_suffix": "Math"},
    {"name": "Perplexity on Code (Github)", "data": data_code, "file_suffix": "Code"},
    {"name": "Perplexity on Book (PG19)", "data": data_book, "file_suffix": "Book"}
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
    ax.set_ylabel("Top-1% Hardest")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # Force scientific notation on Y-axis
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False, fontsize=12)
    
    plt.tight_layout()
    
    # Construct filename
    filename = f"analysis/block_ppl_{suffix}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    output_files.append(filename)
    plt.close(fig)

print(f"Saved files: {output_files}")