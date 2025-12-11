import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from spotlight import get_monkey_patch
from datetime import timedelta

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    else:
        return 0, 0, 1

def build_chat(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
    )

def get_pred_distributed(
        args,
        tokenizer, 
        model,
        data_all, 
        prompt_format,  
        base_out_path, # 注意这里传入的是基础路径
        local_rank,
        rank,
        world_size
):
    # --- 1. 每个 Rank 独立生成自己的文件名 ---
    # 例如: pred/model-method/qasper_rank_0.jsonl
    file_name = os.path.basename(base_out_path) # qasper.jsonl
    dir_name = os.path.dirname(base_out_path)
    my_out_path = os.path.join(dir_name, file_name.replace(".jsonl", f"_rank_{rank}.jsonl"))

    # --- 2. 断点续测 (只检查自己的文件) ---
    done_local_count = 0
    if os.path.exists(my_out_path):
        with open(my_out_path, "r", encoding="utf-8") as f:
            for _ in f:
                done_local_count += 1
    
    # --- 3. 计算本 Rank 应该处理的数据切片 ---
    # 数据分配逻辑：0, 8, 16... 给 Rank 0;  1, 9, 17... 给 Rank 1
    # 所以我应该处理的 indices 是 range(rank, total, world_size)
    # 如果我已经处理了 done_local_count 个，说明前 done_local_count 个已经做完了
    
    my_indices = list(range(rank, len(data_all), world_size))
    # 跳过已经做完的
    remaining_indices = my_indices[done_local_count:]

    if not remaining_indices:
        print(f"Rank {rank} finished all tasks.")
        return

    # 进度条只在 Rank 0 显示简单的，或者每个 Rank 都不显示以免刷屏
    # 这里建议只打印日志
    if rank == 0:
        print(f"Rank 0 starting from index {remaining_indices[0]} (global index)")

    for global_idx in tqdm(remaining_indices, disable=(rank!=0)):
        json_obj = data_all[global_idx]
        prompt = prompt_format.format(**json_obj)
        
        input_ids = build_chat(tokenizer, prompt).to(model.device)
        context_length = input_ids.shape[-1]
        max_new_tokens = model.config.max_position_embeddings - context_length
        if max_new_tokens <= 0: max_new_tokens = 100

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1
            ).ravel().tolist()

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        cot_length = 0
        if '</think>' in pred:
            index = pred.index('</think>')
            cot_content = pred[:index]
            cot_length = len(cot_content)
            pred = pred[index:].replace("</think>", "")
        pred = pred.lstrip()

        # --- 4. 直接写入自己的文件 (No Communication) ---
        res = {
            "pred": pred, 
            "answers": json_obj["answers"], 
            "all_classes": json_obj["all_classes"], 
            "length": json_obj["length"],
            "cot_length": cot_length,
            "index": global_idx # 建议记录 index 方便后续排序
        }
        
        with open(my_out_path, "a", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
            f.write('\n')

    # 确保所有数据写完
    print(f"Rank {rank} finished.")


# --- 新增：合并文件的函数 ---
def merge_results(base_out_path, world_size):
    print(f"Merging results into {base_out_path}...")
    all_data = []
    
    # 读取所有分片
    for r in range(world_size):
        file_name = os.path.basename(base_out_path)
        dir_name = os.path.dirname(base_out_path)
        part_path = os.path.join(dir_name, file_name.replace(".jsonl", f"_rank_{r}.jsonl"))
        
        if os.path.exists(part_path):
            with open(part_path, 'r', encoding='utf-8') as f:
                for line in f:
                    all_data.append(json.loads(line))
            # 可选：合并后删除分片
            # os.remove(part_path) 
    
    # 按照 index 排序 (恢复原始顺序)
    if all_data and "index" in all_data[0]:
        all_data.sort(key=lambda x: x["index"])
        # 如果不需要 index 字段出现在最终结果，可以在这里 pop 掉
        # for d in all_data: d.pop("index", None)
    
    # 写入最终文件
    with open(base_out_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print("Merge finished.")


if __name__ == '__main__':
    local_rank, rank, world_size = setup_distributed()
    
    # ... (参数解析代码保持不变) ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()
    
    # ... (加载模型代码保持不变) ...
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={'': local_rank},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = get_monkey_patch(args.method)(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    datasets_list = ["repobench-p"]
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))

    save_dir_name = f"{args.model_name_or_path.split('/')[-1]}-{args.method}"
    
    for dataset in datasets_list:
        if rank == 0: print(f"Processing {dataset}...")
        if world_size > 1: dist.barrier()

        data = load_dataset('LongBench/LongBench.py', dataset, split='test')
        data_all = [x for x in data]

        save_path = f"pred/{save_dir_name}"
        if rank == 0: os.makedirs(save_path, exist_ok=True)
        if world_size > 1: dist.barrier()
        
        target_out_path = f"{save_path}/{dataset}.jsonl"

        # 执行推理
        get_pred_distributed(
            args, tokenizer, model, data_all, 
            dataset2prompt[dataset], 
            target_out_path, 
            local_rank, rank, world_size
        )
        
        # 等待所有人做完
        if world_size > 1:
            dist.barrier()

        # 由 Rank 0 进行合并
        if rank == 0:
            merge_results(target_out_path, world_size)
        
        if world_size > 1:
            dist.barrier()
            
    if world_size > 1:
        dist.destroy_process_group()