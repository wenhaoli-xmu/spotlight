from spotlight import get_monkey_patch
from spotlight.misc import Evaluator, get_env_conf
import argparse
import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--save-results", action='store_true')
    parser.add_argument("--check-results", action='store_true') 
    parser.add_argument("--method", type=str, default=None)
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

    result = evaluator.evaluate(return_raw=args.save_results or args.check_results)

    if args.save_results:
        model_name = os.path.basename(os.path.normpath(args.model_name_or_path)).lower()
        torch.save({
            "config": test_conf,
            "result": result}, 
            f"test_ppl/{model_name}.pth")

    if args.check_results:
        import IPython
        IPython.embed(header='check results')
