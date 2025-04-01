from abc import abstractmethod
import torch
import json
import logging


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


class Modifier(torch.nn.Module):
    def __init__(self, model, save_ckp, load_ckp):
        super().__init__()
        self.model = model
        self.load_ckp = load_ckp
        self.save_ckp = save_ckp

        if self.load_ckp is not None:
            self.load_checkpoint()

    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def get_conf(self, config):
        if config is not None:
            with open(config, 'r') as f:
                conf = json.load(f)
            print("=" * 40 + " Config " + "=" * 40)
            print(json.dumps(conf, indent=4
                ).replace("\n    ", "\n"
                ).replace("{", ""
                ).replace("}", ""
                ).strip().replace('"', ''))
            print("=" * 88)
            self.conf = conf
        else:
            self.conf = None
    

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad_(False)


    def unfreeze_model(self):
        for param in self.ft_params():
            param.requires_grad_(True)
    

    def load_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.load_ckp
        checkpoint = torch.load(ckp, map_location="cpu")
        try:
            assert len(checkpoint) == len(self.ft_params()), f"number of parameters in checkpoint and model are different"
        except:
            import IPython
            IPython.embed(header='debug')
        for param1, param2 in zip(self.ft_params(), checkpoint):
            param1.data = param2.data.to(device=param1.data.device, dtype=param1.data.dtype)


    def save_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.save_ckp
        torch.save([maybe_zero_3(param) for param in self.ft_params()], ckp)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


    @abstractmethod
    def ft_params(self):
        raise NotImplementedError


    @abstractmethod
    def reset(self):
        raise NotImplementedError
