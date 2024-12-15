import torch, os, json
import argparse
from tokenmix2.misc import get_env_conf


parser = argparse.ArgumentParser()
parser.add_argument("--env_conf", type=str, default=None)
args = parser.parse_args()

json_name = args.env_conf.split('/')[-1]
pth_name = json_name.replace("json", "pth")

results = [None for _ in range(32)]

base_dir = os.path.join("train_results", json_name)
for file in os.listdir(base_dir):
    layer_idx = int(file.split('.')[0])
    results[layer_idx] = torch.load(os.path.join(base_dir, file))


filtered_results = []

for x in results:
    if isinstance(x, (list, tuple)):
        filtered_results += x
    else:
        filtered_results.append(x)

if not os.path.exists('ckp'):
    os.mkdir('ckp')

torch.save(filtered_results, os.path.join("ckp", pth_name))
print(f"done!")
