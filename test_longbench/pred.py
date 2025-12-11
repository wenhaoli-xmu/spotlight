import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import argparse
from spotlight import get_monkey_patch
import json, os

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch.distributed as dist


def build_chat(tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors='pt')


def get_pred(
        args,
        tokenizer, 
        model,
        data, 
        prompt_format,  
        out_path,
):
    start_idx = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for _ in f:
                start_idx += 1
    
    data = data[start_idx:]
    if len(data) == 0:
        return

    for json_obj in tqdm(data):

        prompt = prompt_format.format(**json_obj)
        input_ids = build_chat(tokenizer, prompt).to('cuda:0')
        context_length = input_ids.shape[-1]

        max_new_tokens = model.config.max_position_embeddings - input_ids.shape[-1]

        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        ).ravel().tolist()

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        cot_length = 0
        
        # step-1: remove think content
        if '</think>' in pred:
            index = pred.index('</think>')
            cot_content = pred[:index]
            cot_length = len(cot_content)
            pred = pred[index:].replace("</think>", "")
        
        # step-2: remove meaningless tokens
        pred = pred.lstrip()

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred, 
                "answers": json_obj["answers"], 
                "all_classes": json_obj["all_classes"], 
                "length": json_obj["length"],
                "cot_length": cot_length  # 新增: 保存 CoT 长度
            }, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    datasets = [
        # "narrativeqa", 
        "qasper", 
        # "multifieldqa_en", 
        # "multifieldqa_zh", 
        # "hotpotqa", 
        # "2wikimqa", 
        # "musique", 
        # "dureader", 
        # "gov_report", 
        # "multi_news", 
        "qmsum", 
        # "vcsum", 
        # "trec", 
        # "triviaqa", 
        # "samsum", 
        # "lsht", 
        # "passage_count", 
        # "passage_retrieval_en", 
        # "passage_retrieval_zh", 
        # "lcc", 
        "repobench-p"
    ]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        torch_dtype=torch.bfloat16)
    model = get_monkey_patch(args.method)(model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    save_path = f"{args.model_name_or_path.split('/')[-1]}-{args.method}"

    for dataset in datasets:
        data = load_dataset('LongBench/LongBench.py', dataset, split='test')
        if not os.path.exists(f"pred/{save_path}"):
            os.makedirs(f"pred/{save_path}")
        out_path = f"pred/{save_path}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        data_all = [data_sample for data_sample in data]

        get_pred(
            args,
            tokenizer,
            model,
            data_all, 
            prompt_format, 
            out_path)