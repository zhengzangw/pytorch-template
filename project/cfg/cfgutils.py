import argparse
import collections
import json
import os

import numpy as np
import torch
import yaml

__all__ = [
    "load_config",
    "save_config",
    "flatten_dict",
    "sanitize_dict",
    "update_namespace",
    "extract",
    "s2b",
    "g",
]

# Load config file
def load_yaml(f_path):
    with open(f_path, "r") as stream:
        return yaml.safe_load(stream)


def load_json(f_path):
    with open(f_path, "r") as f:
        return json.load(f)


def load_config(path, flatten=True):
    _, ext = os.path.splitext(path)

    assert ext in [
        ".json",
        ".yaml",
        ".yml",
    ], f"Only support yaml and json config, but '{ext}' given."
    if ext == "json":
        cfg = load_json(path)
    else:
        cfg = load_yaml(path)

    if cfg is None:
        cfg = dict()

    if flatten:
        cfg = flatten_dict(cfg)
    return cfg


# Dump config file
def save_json(obj, f_path):
    with open(f_path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def save_yaml(obj, f_path):
    with open(f_path, "w") as f:
        yaml.dump(obj, f)


def save_config(obj, path, ext=None):
    _, fext = os.path.splitext(path)
    if fext.startswith("."):
        fext = fext[1:]
    if fext != "":
        assert (
            ext == None or fext == ext
        ), f"Extension conflict between '{path}' and '{ext}'."
        ext = fext

    if ext in ["yaml", "yml"]:
        save_yaml(obj, path)
    else:
        save_json(obj, path)


# Utils
def flatten_dict(d, keep_parent=False, sep="_", parent_key=""):
    """Flatten dict to only one nest

    Args:
        d (dict): dictionary to flatten
        keep_parent (bool, optional): If True, keep parent's key name, and keys should all be str. Defaults to False.
        sep (str, optional): Effective only keep_parent=True, separator between keys. Defaults to "_".
        parent_key (str, optional): For recursive call. Defaults to "".

    Returns:
        dict: flattened dict
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key and keep_parent else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(
                flatten_dict(v, keep_parent, parent_key=new_key, sep=sep).items()
            )
        else:
            items.append((new_key, v))

    items_key = [i[0] for i in items]
    assert len(items_key) == len(set(items_key))

    return dict(items)


def sanitize_dict(params, to_str=True, none_fill="N/A"):
    """Convert all items into tensorboard supported values or str

    Args:
        params (dict): dict to sanitize
        to_str (bool, optional): If True, turn all items to string. Defaults to True.

    Returns:
        dict: sanitized dict
    """
    items = []
    for k in params.keys():
        # numpy to float
        if isinstance(params[k], (np.bool_, np.integer, np.floating)):
            items.append([k, params[k].item()])
        elif isinstance(params[k], np.ndarray):
            items.append([k, str(params[k].tolist())])
        # torch to float
        elif isinstance(params[k], torch.Tensor):
            items.append([k, str(params[k].tolist())])
        # None to str
        elif params[k] is None:
            items.append([k, none_fill])
        # Others to str
        elif type(params[k]) not in [bool, int, float, str, torch.Tensor]:
            items.append([k, str(params[k])])
        else:
            items.append([k, params[k]])

        # All to str
        if to_str:
            items[-1][-1] = str(items[-1][-1])

    return dict(items)


def update_namespace(args, dictionary, overwrite=True, rest=False):
    """update Namespace with given dictionary

    Args:
        args (Namespace): Namespace to be updated
        dictionary (dict): dictionary
        overwrite (bool, optional): If True, All Namespace value will overwritten by dictionary value. Otherwise, only Namespace with None will be overwritten. Defaults to True.
        rest: Effective only if overwrite=True. If True, add keys in dictionary but not in args into args. Otherwise raise an error.

    Returns:
        Namespace
    """
    dict_args = vars(args)

    if overwrite:
        dict_args.update(dictionary)
    else:
        for k, v in dict_args.items():
            if v is not None:
                pass
            elif k in dictionary:
                dict_args[k] = dictionary[k]
        for k, v in dictionary.items():
            if k not in dict_args:
                if rest:
                    dict_args[k] = v
                else:
                    raise KeyError(f"no key {k}")

    args = argparse.Namespace(**dict_args)
    return args


def extract(s, delimit="-", num=0):
    """Extract the num_th word from string s

    Args:
        s (str): string to be parsed
        delimit (str, optional): delimiter. Defaults to "-".
        num (int, optional): . Defaults to 0.

    Returns:
        (str, List[str])
    """
    s_list = s.split(delimit)
    first = s_list[num]
    s_list.pop(num)
    s_rest = delimit.join(s_list)
    return first, s_rest


# argparse type
def s2b(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# template generator for params.py
def g(template, name_list, placeholder="{}"):
    items = []
    for name in name_list:
        t = []
        t.append(template[0].replace(placeholder, name))
        t.append(template[1].replace(placeholder, name))
        t.extend(template[2:])
        items.append(t)
    return items
