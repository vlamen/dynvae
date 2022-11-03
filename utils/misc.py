import torch
import sys
from collections import UserDict
import warnings


def new_tensor(obj, dtype=torch.float):
    """
    Make a new tensor out of obj
    :param obj: tensor or tensor-like
    :param dtype: target dtype
    :return: new torch.Tensor that does not require gradients from obj
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach().type(dtype)
    else:
        return torch.as_tensor(obj, dtype=dtype)


def load_model(path: str, architecture: type, to_cpu=True, **kwargs):
    model = architecture(**kwargs)
    device = torch.device('cpu') if to_cpu else None
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        warnings.warn(f'\nCould not load model from {path}. \nEncountered the following runtime error:\n{e}\nreturning None\n', stacklevel=2)
        model = None
    return model


class MyWriter:  # this was a terrible monkey patch
    def __init__(self, path):
        self.content = []
        self.path = path 

    def flush(self):
        with open(self.path, 'a+') as file:
            file.writelines(self.content)
            self.content = []
    
    def write(self, line):
        sys.stdout.write(line)
        self.content.append(line)

    def __del__(self):
        self.flush()


class DictList(UserDict):
    def __init__(self, initial_data=None, /, **kwargs):
        if initial_data is not None:
            raise ValueError('Initial data not supported for DictList')
        super().__init__()

    @staticmethod
    def split_key(key: str):
        k, i = key.split('-')
        key = k 
        index = int(i)
        return key, index 

    @staticmethod
    def join_key(key, index):
        return f'{key}-{index}'

    def __getitem__(self, key):
        if isinstance(key, tuple):
            k, i = key
            return self.data[k][i]
        elif '-' in key:
            k, i = self.split_key(key)
            return self.data[k][i]
        else:
            return self.data[key]
    
    def __setitem__(self, key, item):
        if isinstance(key, tuple):
            k, i = key 
            self.data[k][i] = item 
        elif '-' in key:
            k, i = self.split_key(key)
            self.data[k][i] = item 
        else:
            if key not in self.data:
                self.data[key] = [item]
            else:
                self.data[key].append(item)

    def get_flat_dict(self):
        flat_dict = {}
        for k, l in self.data.items():
            for i, v in enumerate(l):
                flat_dict[self.join_key(k, i)] = v 
        return flat_dict
    
    def __str__(self):
        return str(self.get_flat_dict())
    
    def __repr__(self):
        return repr(self.get_flat_dict())

    def keys(self):
        key_list = []
        for k, l in self.data.items():
            for i in range(len(l)):
                key_list.append(self.join_key(k, i))
        return key_list

    def base_keys(self):
        return self.data.keys()
    
    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return ((key, self[key]) for key in self.keys())

    def values(self):
        return (item[1] for item in self.items())

    def __len__(self):
        l = 0
        for val_list in self.data.values():
            l += len(val_list)

    def __delitem__(self, key):
        raise NotImplementedError(
            "DictList does not implement deletion of items directly. Deletion can be accomplished via self.data"
        )
    
    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple):
            k, i = key 
            if k not in self.data:
                return False 
            return len(self[k]) > i 
        elif '-' in key:
            return self.__contains__(self.split_key(key))
        else: 
            return key in self.data
