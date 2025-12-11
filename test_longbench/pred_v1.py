import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from spotlight.misc import get_model_and_tokenizer


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt):

    import IPython
    IPython.embed(header='check template')


# modified: 将参数max_length去掉
def get_pred(
        env_conf, 
        tokenizer, 
        model,
        data, 
        max_gen, 
        prompt_format, 
        dataset, 
        device, 
        model_name, 
        out_path, 
        model_max_length,
        chat_template,
        magicpig: bool = False):

    
    for json_obj in tqdm(data):

        prompt = prompt_format.format(**json_obj)
        prompt = build_chat(tokenizer, prompt)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        # # ======================================================================
        # # NOTE: right truncation
        # if input.input_ids.shape[-1] >= model_max_length - max_gen:
        #     input.input_ids = input.input_ids[..., -model_max_length + max_gen:]
        # context_length = input.input_ids.shape[-1]
        # # ======================================================================


        # ============================================================================
        # NOTE: 新增加
        if magicpig:
            prompt = tokenizer.decode(input.input_ids.ravel().tolist())
            pred = model(prompt, max_new_tokens=max_gen)['text'][0]
        else:
            output = model.generate(
                input_ids=input.input_ids,
                max_new_tokens=max_gen
            ).ravel().tolist()

            # NOTE: 新增加
            if tokenizer.eos_token_id in output:
                index = output.index(tokenizer.eos_token_id)
                output = output[:index]
            torch.cuda.empty_cache()

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        # ============================================================================


        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument("--max_gen", type=int, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--time_stamp', type=str)
    args = parser.parse_args()

    import json, os
    with open(args.env_conf, "r") as f:
        env_conf = json.load(f)
    with open("test_longbench/pred.json", 'r') as f:
        pred_conf = json.load(f)

    seed_everything(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.env_conf.split('/')[-1]

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    # NOTE: 加载数据集
    for dataset in pred_conf:
        assert dataset in datasets
    datasets = pred_conf

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    # load model
    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])

    for dataset in datasets:

        get_pred(
            env_conf, 
            tokenizer,
            model,
            data_all, 
            max_gen if args.max_gen is None else args.max_gen, 
            prompt_format, 
            dataset, 
            device, 
            model_name, 
            out_path, 
            args.model_max_length, 
            args.chat_template,
            args.magicpig)
