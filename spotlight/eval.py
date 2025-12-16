from itertools import chain
from .data import get_corpus
from lm_eval.metrics import perplexity
from torch.utils.data import DataLoader
import torch
import tqdm

import numpy as np
from .io_wrapper import TestIOWrapper

from lm_eval.metrics import perplexity
from itertools import chain
import numpy as np

def post_process(accum_total_output, task_type):
    for i in range(len(accum_total_output)):
        if task_type == "auto encoding":
            accum_total_output[i] = accum_total_output[i][0]
        elif task_type == 'ppl':
            output = accum_total_output[i]
            output = list(chain.from_iterable(output))
            output = perplexity(output)
            accum_total_output[i] = output
        elif task_type == 'top_ppl':
            output = accum_total_output[i]
            output = list(chain.from_iterable(output))
            output = perplexity(output)
            accum_total_output[i] = output
        elif task_type == 'bottom_ppl':
            output = accum_total_output[i]
            output = list(chain.from_iterable(output))
            output = perplexity(output)
            accum_total_output[i] = output
        elif task_type == 'dist_ppl':
            output = accum_total_output[i]
            output = list(chain.from_iterable(output))
            output = np.histogram(np.array([-x for x in output]), bins=1000, range=[0, 1])
            output = np.cumsum(output)
            output /= output[-1]
            accum_total_output[i] = output

    return accum_total_output


@torch.no_grad()
def test_on_task(model, tokenizer, task_type, task_name, num_instance, truncation, callback=None):

    io_wrapper = TestIOWrapper(tokenizer, truncation)
    task = get_corpus(task_name)
    loader = iter(DataLoader(task, batch_size=1, shuffle=False))
    accum_total_output = []

    for _ in range(num_instance):
        data = next(loader)
        assert "task_type" in data.keys()
        data["task_type"] = task_type

        total_output = []

        inputs, compute_loss = io_wrapper.wrap(data)

        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].cuda()

        outputs = model(**inputs)

        if compute_loss is not None:
            result = compute_loss(outputs)
            total_output.append(result)

        accum_total_output.append(total_output)

        if hasattr(model, 'reset'):
            model.reset()
        
        if callback is not None:
            assert callable(callback)
            callback(outputs)

        torch.cuda.empty_cache()

    if task_type != 'return_raw':
        accum_total_output = post_process(accum_total_output, task_type)
        result = {
            "task_name": task_name,
            "task_type": task_type,
            "num_instance": num_instance,
            "truncation": truncation,
            "avg": float(np.mean(accum_total_output)),
            "max": float(max(accum_total_output)),
            "min": float(min(accum_total_output))
        }
    else:
        result = accum_total_output

    return result
