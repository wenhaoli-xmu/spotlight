from spotlight import get_monkey_patch
from spotlight.misc import Evaluator, get_env_conf
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--dist-ppl", action='store_true')
    args = parser.parse_args()

    test_conf = get_env_conf("test_ppl/test.json")
    monkey_patch = get_monkey_patch(args.method)
    print('config loaded ✅', flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        torch_dtype=torch.bfloat16)
    model = monkey_patch(model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    evaluator_class = Evaluator
    evaluator = evaluator_class(model, tokenizer, eval=None, tasks=test_conf)

    result = evaluator.evaluate(return_raw=args.dist_ppl)

    if args.dist_ppl:
        raise NotImplementedError
