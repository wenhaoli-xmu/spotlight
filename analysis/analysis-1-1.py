import os
import json
import torch

DATA_COUNT = 7
LAYER_COUNT = 24
TOPK = 0.05
CACHE = 0.05
SUB_DIR = 'gpt-oss-20b'

for data_idx in range(DATA_COUNT):
    root_dir = f'/mnt/petrelfs/liwenhao/spotlight/attns/{SUB_DIR}/data-{data_idx}'

    rate_avgs = []
    rate_avgs2 = []

    for layer_idx in range(LAYER_COUNT):
        path = os.path.join(root_dir, f'layer-{layer_idx}.pth')

        t_list = torch.load(path, map_location='cpu')
        max_length = t_list[-1].numel()

        

        # step-1 将所有的map成为 topk indices
        budget = int(t_list[0].numel() * TOPK)
        t_list = [torch.topk(t, k=budget).indices.tolist() for t in t_list]
        s_list = [set(t) for t in t_list]

        rate = []
        rate2 = []

        # step-2 计算留下来的比率
        for i in range(len(s_list) - 1):
            x, y = s_list[i], s_list[i+1]
            i_set = x & y
            rate.append(len(i_set) / len(x))

            z = set((xx + 1 for xx in x))
            i_set = (x | z) & y
            rate2.append(len(i_set) / len(x))

        if len(rate) > 0:
            rate_avgs.append(sum(rate) / len(rate))
            rate_avgs2.append(sum(rate2) / len(rate2))

    path = os.path.join(root_dir, "summary.json")
    result = {f"layer-{i}": rate_avgs[i] for i in range(LAYER_COUNT)}
    result.update({"avg": sum(rate_avgs) / len(rate_avgs)})
    
    with open(path, 'w') as f:
        json.dump(result, f, indent=4)

    path = os.path.join(root_dir, "summary2.json")
    result = {f"layer-{i}": rate_avgs2[i] for i in range(LAYER_COUNT)}
    result.update({"avg": sum(rate_avgs2) / len(rate_avgs2)})

    with open(path, 'w') as f:
        json.dump(result, f, indent=4)
