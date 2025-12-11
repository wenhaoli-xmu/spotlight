import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
import torch

from crop import crop
import sys
sys.path.append("/home/lwh/token-mix-2")
from spotlight.misc import get_model_and_tokenizer, get_env_conf

openai.api_key = "INSERTYOURKEYHERE"
choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, dev_df, test_df, model, tokenizer):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]


        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).input_ids
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits = model.forward(input_ids).logits
        logits = logits[:,-1:,:]

        answer_list = ["{}".format(ans) for ans in answers]
        index = torch.tensor([tokenizer(ans, add_special_tokens=False).input_ids[0] for ans in answer_list], dtype=torch.int64)

        logits = logits[..., index].ravel().tolist()
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(logits)]
        probs = softmax(np.array(logits))

        # lprobs = []
        # for ans in answers:
        #     try:
        #         lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
        #     except:
        #         print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
        #         lprobs.append(-100)


        # pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        # probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):

    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])

    ckp_file = env_conf['model']['save_ckp']
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dose not exists")


    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print(subjects)
    print(args)

    all_cors = []

    engine = "davinci"
    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, dev_df, test_df, model, tokenizer)
        all_cors.append(cors)

        test_df["{}_correct".format(engine)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str)
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="mmlu/data")
    parser.add_argument("--save_dir", "-s", type=str, default="mmlu/output")
    args = parser.parse_args()

    main(args)

