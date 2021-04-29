from .cfgutils import *

params = [
    ("config", "config to load (will be overwritten by args)", str),
    ("load_from_checkpoint", "checkpoint to load", str),
    # program_params:
    ("name", "name of experiments"),
    ("seed", "random seed", int),
    ("n_class", "n_class", int),
    # epoch_params:
    # logger_params:
    # augmentation_params:
    ("size", "resize images"),
    ("augmentations", "augs"),
    # dataset_params:
    ("dataset", "name of dataset", str),
    # model_params:
    ("model", "name of model"),
    ("model_args", "dict, args for model"),
    ("criterion", "criterion function to use", str),
    # loss_params:
    # optim_params:
    ("optimizer", "which optimizer to use", str),
    ("scheduler", "which schduler to use", str),
    ("scheduler_args", "dict, args for schedular"),
    ("warmup", "if use warmup", bool),
    ("lr", "learning rate", float),
    # train_params:
    ("batch_size", "batch size (bs = num_gpus * batch_size if DDP)", int),
    ("num_workers", "number of workers to use for data loader", int),
]


def set_dynamic_default(args):
    if args.model_args is None:
        args.model_args = dict()
    return args
