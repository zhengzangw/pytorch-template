import argparse

from .cfgutils import *
from .params import params, set_dynamic_default

__all__ = ["get_parser", "parse_args"]


def argparse_one_entry(parser, name, help_text="", type_text=str, default=None):
    """Parser for python params
    """
    if isinstance(type_text, tuple) and type_text[0] == list:
        parser.add_argument(
            f"--{name}", help=help_text, nargs="+", type=type_text[1], default=default
        )
    else:
        if type_text == bool:
            type_text = s2b
        parser.add_argument(
            f"--{name}", help=help_text, type=type_text, default=default
        )


def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help)
    return parser


def auto_parse(parser, custom_params=None):
    if custom_params is not None:
        params_ = custom_params
    else:
        params_ = params

    # parse entries in params.py
    for entry in params_:
        argparse_one_entry(parser, *entry)

    return parser


def parse_args(
    parser=None,
    args=None,
    config=True,
    custom_set_dynamic_default=None,
    is_auto_parse=True,
):
    # parser
    if parser is None:
        parser = get_parser()

    if is_auto_parse:
        auto_parse(parser)

    # parse
    args, unknown = parser.parse_known_args(args)
    if len(unknown):
        print(f"Ignore unknown args: {unknown}")

    # config
    if config and args.config:
        dict_cfg = load_config(args.config, flatten=False)
        args = update_namespace(args, dict_cfg, overwrite=False)

    # dynamic default
    if custom_set_dynamic_default is not None:
        set_dynamic_default_ = custom_set_dynamic_default
    else:
        set_dynamic_default_ = set_dynamic_default
    args = set_dynamic_default_(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    # parser = get_parser()
    # args = parse_args(parser)
    breakpoint()
