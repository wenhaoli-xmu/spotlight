import torch, os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--env_conf", type=str, default=None)
parser.add_argument("--num_layers", type=int, default=32)
args = parser.parse_args()

json_name = args.env_conf.split('/')[-1]
pth_name = json_name.replace("json", "pth")

results = [None for _ in range(args.num_layers)]

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
