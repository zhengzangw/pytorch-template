from .cfgutils import *

params = [
    ("config", "config to load (will be overwritten by args)", str),
    ("load_from_checkpoint", "", str),
    # program_params:
    ("name", "name of experiments"),
    ("seed", "random seed", int),
    ("src", "source dataset", str),
    ("tgt", "target dataset", str),
    ("n_class", "n_class", str),
    # epoch_params:
    # logger_params:
    # augmentation_params:
    ("size", "resize images"),
    ("augmentations", "augs"),
    # dataset_params:
    # model_params:
    ("model", "name of model"),
    ("model_args", "dict"),
    ("criterion", ""),
    ("load_weight", "", bool),
    # loss_params:
    # optim_params:
    ("optimizer", "", str),
    ("lr", "learning rate", float),
    ("poly_decay", "", bool),
    ("warmup_lr", "", float),
    # train_params:
    ("batch_size", "batch size", int),
    ("num_workers", "number of workers to use for data loader", int),
    ("superpixel", "", bool),
]


def set_dynamic_default(args):
    if args.model_args is None:
        args.model_args = dict()
    return args
