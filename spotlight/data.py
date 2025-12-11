import json
from torch.utils.data import Dataset
from datasets import load_dataset



def pack_lm(text: str):
    assert isinstance(text, str)
    return {
        "text": text,
        "task_type": "language modeling"
    }
    

class PG19(Dataset):
    """
    pg19.train
    pg19.test
    pg19.train.1m
    pg19.test.1m
    pg19.train.256k
    pg19.test.256k
    pg19.train.128k
    pg19.test.128k
    """

    def __init__(self, split, max_length=None):
        import os
        assert os.path.exists(os.environ['SPOTLIGHT_PG19_PATH']), f"Please download pg19.json first."
        with open(os.environ['SPOTLIGHT_PG19_PATH'], "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]
        
        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        elif max_length == '128k':
            self.maximum = 128 * 1024
        else:
            self.maximum = None

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']

        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)
    

class ProofPile(Dataset):
    """
    proof-pile
    proof-pile.1m
    proof-pile.256k
    """

    def __init__(self, max_length=None):
        import os
        assert os.path.exists(os.environ['SPOTLIGHT_PROOFPILE_PATH']), f"Please download proof-pile.json first."
        with open(os.environ['SPOTLIGHT_PROOFPILE_PATH'], "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        else:
            self.maximum = None

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']
        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)


class CodeParrot(Dataset):
    """
    code-parrot
    code-parrot.1m
    code-parrot.256k
    code-parrot.128k
    """

    def __init__(self, max_length=None):
        import os
        assert os.path.exists(os.environ['SPOTLIGHT_CODEPARROT_PATH']), f"Please download codeparrot.json first."
        with open(os.environ['SPOTLIGHT_CODEPARROT_PATH'], "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        elif max_length == '128k':
            self.maximum = 128 * 1024
        elif max_length is None:
            pass
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']
        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)


CORPUS_MAPPING = {
    # language modeling
    "pg19": PG19,
    "proof-pile": ProofPile,
    "code-parrot": CodeParrot
}


def get_corpus(ds):
    ds = ds.lower().replace(" ", "")
    dataset_name, *args = ds.split(".")

    for name, data_class in CORPUS_MAPPING.items():
        if name == dataset_name:
            return data_class(*args)
    
    raise NotImplementedError(dataset_name)
