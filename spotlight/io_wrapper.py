import types
from abc import ABC
from dataclasses import dataclass
from typing import Any
from functools import partial

import torch


def drop_tuple(x):
    return x[0] if isinstance(x, (tuple, list)) else x


@dataclass
class WrapperOutput:
    inputs: dict = None
    compute_loss: types.FunctionType = None

    def __getitem__(self, idx):
        return (
            self.inputs,
            self.compute_loss,
        )[idx]
    

class BasicIOWrapper(ABC):
    def __init__(self, tokenizer, truncation):
        self.tokenizer = tokenizer
        self.task_type = None
        self.truncation = truncation


    def wrap(self, data: dict) -> Any:
        self.task_type = drop_tuple(data["task_type"])

        if self.task_type == 'perplexity':
            text = drop_tuple(data["text"])
            return self.wrap_ppl_task(text, self.truncation)

        else:
            raise NotImplementedError(self.task_type)
        

class TestIOWrapper(BasicIOWrapper):
    def wrap_ppl_task(self, text: str, truncation: int):
        text = self.tokenizer(text, truncation=False, return_tensors='pt')
        input_ids = text.input_ids[:,:truncation]

        def compute_ppl(outputs, input_ids):
            logits = outputs.logits.cpu().log_softmax(dim=-1)
            gold_indices = input_ids[:,-logits.shape[1] + 1:].cpu()
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            return logprobs[1:]
        
        ret = WrapperOutput(
            inputs={"input_ids": input_ids},
            compute_loss=partial(compute_ppl, input_ids=input_ids))
        return ret
