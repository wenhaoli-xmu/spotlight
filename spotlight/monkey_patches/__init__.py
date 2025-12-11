import os
import json
from functools import partial


def get_config(method):
    config_file = os.path.join("config", f"{method}.json")
    with open(config_file, 'r') as f:
        return json.load(f)


def get_monkey_patch(method):

    if method == "origin":
        from .origin import monkey_patch
    
    elif method == 'topk-eval':
        from .topk_eval import monkey_patch
    
    elif method == 'topk-gen':
        from .topk_gen import monkey_patch

    elif method == 'topk-block-eval':
        from .topk_block_eval import monkey_patch
    
    elif method == 'hash-train':
        from .hash_train import monkey_patch

    elif method == 'hash-eval':
        from .hash_eval import monkey_patch

    return partial(monkey_patch, config=get_config(method))