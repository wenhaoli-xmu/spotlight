from ..modifier import Modifier
import torch
from profiler import WallTime


class Greedy(Modifier):
    def __init__(self, model, save_ckp=None, load_ckp=None, config=None):
        super().__init__(model, save_ckp, load_ckp)


    def ft_params(self):
        return []

    
    def reset(self):
        pass


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2]):

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)

        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        # prefilling
        output = self.model(input_ids=input_ids, use_cache=True)
        logits, kv_cache = output.logits, output.past_key_values
        new_tok = logits[:, -1:].argmax(dim=-1)
        new_ids = []

        while len(new_ids) < max_new_tokens:
            if new_tok.ravel().item() in eos_token_id: break
            new_ids.append(new_tok.ravel().item())
            
            output = self.model(input_ids=new_tok, past_key_values=kv_cache, use_cache=True)
            logits, kv_cache = output.logits, output.past_key_values
            new_tok = logits.argmax(dim=-1)

        self.model.reset()
        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
        return torch.cat([input_ids, new_ids], dim=-1)
