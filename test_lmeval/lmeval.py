from lm_eval.base import BaseLM
from lm_eval import evaluator
from spotlight.misc import get_model_and_tokenizer, get_env_conf
import argparse
import torch
import os
import json



class MyModel(BaseLM):
    def __init__(self, model, tokenizer, model_max_length):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length


    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    

    @property
    def max_length(self):
        return self.model_max_length


    @property
    def max_gen_toks(self):
        return 256
    

    @property
    def batch_size(self):
        return 1
    

    @property
    def device(self):
        device = next(self.model.parameters()).device
        return device
    
    
    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
    

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps).logits
        

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--fewshot", type=int, default=5)

    # Arguments for MagicPIG (https://arxiv.org/abs/2410.16179)
    parser.parse_args('--magicpig', action='store_true')
    args = parser.parse_args()


    with open("test_lmeval/lmeval.json", 'r') as f:
        tasks = json.load(f)

    env_conf = get_env_conf(args.env_conf)

    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    model.eval()
    ckp_file = env_conf['model']['save_ckp']
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dose not exists")

    adapter = MyModel(model, tokenizer, args.model_max_length)
    result = evaluator.simple_evaluate(
        model=adapter,
        num_fewshot=args.fewshot,
        batch_size=1,
        tasks=tasks,
        limit=args.limit,
        no_cache=True)
    
    print(json.dumps(result['results'], indent=4))