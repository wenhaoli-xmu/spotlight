# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import json
import yaml
import os
import sys
import importlib
import time
from tqdm import tqdm
from typing import Optional

SERVER_TYPES = (
    'trtllm',
    'vllm',
    'openai',
    'gemini',
    'hf',
    'mamba',
)

from dataclasses import dataclass, field

@dataclass
class MagicpigConfig:
    server_type: str = 'hf'
    server_host: str = '127.0.0.1'
    server_port: str = '5000'
    ssh_server: Optional[str] = None
    ssh_key_path: Optional[str] = None
    model_name_or_path: str = 'meta-llama/Llama-2-7b-chat-hf'

    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 1.0
    random_seed: int = 0
    stop_words: list = field(default_factory=list)
    sliding_window_size: int = None
    threads: int = 1
    
    K: int = 10
    L: int = 150
    S: float = 4.0
    W: int = 64
    Q: int = 0
    QR: float = 0.0
    max_seq_length: int = 4096
    max_new_tokens: int = 128
    do_sample: bool = True


def get_magicpig(config: MagicpigConfig):
    if config.server_type == 'trtllm':
        from .client_wrappers import TRTLLMClient
        llm = TRTLLMClient(
            server_host=config.server_host,
            server_port=config.server_port,
            ssh_server=config.ssh_server,
            ssh_key_path=config.ssh_key_path,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            random_seed=config.random_seed,
            stop=config.stop_words,
            tokens_to_generate=config.max_new_tokens,
            max_attention_window_size=config.sliding_window_size,
        )

    elif config.server_type == 'vllm':
        from .client_wrappers import VLLMClient
        llm = VLLMClient(
            server_host=config.server_host,
            server_port=config.server_port,
            ssh_server=config.ssh_server,
            ssh_key_path=config.ssh_key_path,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            random_seed=config.random_seed,
            stop=config.stop_words,
            tokens_to_generate=config.max_new_tokens,
        )
        
    elif config.server_type == 'openai':
        from .client_wrappers import OpenAIClient
        llm = OpenAIClient(
            model_name=config.model_name_or_path,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            random_seed=config.random_seed,
            stop=config.stop_words,
            tokens_to_generate=config.max_new_tokens,
        )

    elif config.server_type == 'gemini':
        from .client_wrappers import GeminiClient
        llm = GeminiClient(
            model_name=config.model_name_or_path,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            random_seed=config.random_seed,
            stop=config.stop_words,
            tokens_to_generate=config.max_new_tokens,
        )
        
    elif config.server_type == 'hf':
        from .model_wrappers import HuggingFaceModel
        import torch
        import numpy as np
        seed = 43
        import random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        llm = HuggingFaceModel(
            name_or_path=config.model_name_or_path,
            do_sample=config.temperature > 0,
            repetition_penalty=1,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stop=config.stop_words,
            K = config.K,
            L = config.L,
            W = config.W,
            S = config.S,
            approx=(config.K > 0),
            Q=config.Q,
            QR=config.QR,
            max_new_tokens=config.max_new_tokens,
        )
    
    elif config.server_type == 'mamba':
        from .model_wrappers import MambaModel
        # mamba uses its own generation function, do not pass in do_sample
        # https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/utils/generation.py#L121
        llm = MambaModel(
            name_or_path=config.model_name_or_path,
            repetition_penalty=1,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stop=config.stop_words,
            max_new_tokens=config.max_new_tokens,
        )
        
    else:
        raise RuntimeError(f'Unsupported server type {config.server_type}')

    return llm
