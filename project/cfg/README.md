## Config Management

To custom config files, edit `params` and `set_dynamic_default` in params.py

```py
params = [
    # args (name, help[Optional], type[Optional], default[Optional])
    ("config", "config to load (will be overwritten by args)"),
    # type [str(default), int, float, bool]
    ("loss", "list of loss to apply", (list, str)),
    # *g(args with placeholder ({}), ['a', 'b'])
    *g(("log_{}", "if True, log {}", bool), ("train", "test"),),
]
```

## Conflict on config and command options

If specified in command line options, it will override the options in config flie. However, this is not true for options with default value not `None`. **Thus, default value in params is discouraged!**

Dictionary can be only specified in config files.

## Get Started

```py
import cfg

parser = cfg.get_parser(add_help=True)
parser = pl.Trainer.add_argparse_args(parser) # if used with pl
cfgs = cfg.parse_args(parser)
```
