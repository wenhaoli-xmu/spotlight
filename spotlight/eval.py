from itertools import chain
from .data import get_corpus
from lm_eval.metrics import perplexity
from torch.utils.data import DataLoader
import torch
import tqdm

import numpy as np
from .io_wrapper import (
    TestIOWrapper)

from lm_eval.metrics import perplexity
from itertools import chain


def post_process(accum_total_output, task_type):
    for i in range(len(accum_total_output)):
        if task_type == "auto encoding":
            accum_total_output[i] = accum_total_output[i][0]
        elif task_type == "perplexity":
            output = accum_total_output[i]
            output = list(chain.from_iterable(output))
            output = perplexity(output)
            accum_total_output[i] = output
    
    return accum_total_output


@torch.no_grad()
def test_on_task_for_quest(model, tokenizer, task_type, task_name, num_instance, truncation, callback=None):

    io_wrapper = TestIOWrapper(tokenizer, truncation)
    task = get_corpus(task_name)
    loader = iter(DataLoader(task, batch_size=1, shuffle=False))
    accum_total_output = []

    for _ in tqdm.tqdm(range(num_instance)):
        data = next(loader)
        assert "task_type" in data.keys()
        data["task_type"] = task_type

        total_output = []

        inputs, compute_loss = io_wrapper.wrap(data)

        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].cuda()

        assert isinstance(inputs, dict) and len(inputs) == 1 and 'input_ids' in inputs
        input_ids = inputs['input_ids']
        past_key_values = None
        logits = None

        for token in tqdm.tqdm(input_ids.chunk(input_ids.shape[-1], dim=-1)):
            outputs = model(input_ids=token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits if logits is None else torch.cat([logits, outputs.logits], dim=-2)
        
        outputs.logits = logits

        if compute_loss is not None:
            result = compute_loss(outputs)
            total_output.append(result)

        accum_total_output.append(total_output)

        if hasattr(model, 'reset'):
            model.reset()
        
        if callback is not None:
            assert callable(callback)
            callback(outputs)

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

    return result


@torch.no_grad()
def test_on_task_for_magicpig(model, tokenizer, task_type, task_name, num_instance, truncation, callback=None):

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

        assert isinstance(inputs, dict) and len(inputs) == 1 and 'input_ids' in inputs
        input_ids = inputs['input_ids']
        past_key_values = None
        logits = None


        seq_len = input_ids.shape[-1]
        prefill_len = 128
        fisrt_few_tokens, rest_tokens = input_ids[:, :prefill_len], input_ids[:, prefill_len:]

        model.model.select_kv(True)

        # pre-filling
        outputs = model.model(input_ids=fisrt_few_tokens, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits

        # decoding
        for token in rest_tokens.chunk(seq_len - prefill_len, dim=-1):
            outputs = model.model(input_ids=token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = torch.cat([logits, outputs.logits], dim=-2)

        print(f"sparsity: {sum(past_key_values.history_sparse) / len(past_key_values.history_sparse)}")

        

        outputs.logits = logits

        if compute_loss is not None:
            result = compute_loss(outputs)
            total_output.append(result)

        accum_total_output.append(total_output)

        if hasattr(model, 'reset'):
            model.reset()
        
        if callback is not None:
            assert callable(callback)
            callback(outputs)

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

    return result


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

    return result
