#!/usr/bin/env python3
import os
import shutil

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import DDPPlugin

from project import cfg
from project.dataset import DataModule
from project.lit_project import LitProject
from project.utils import callbacks


def cli_main():
    # ------------
    # args
    # ------------
    parser = cfg.get_parser(add_help=True)
    parser = pl.Trainer.add_argparse_args(parser)
    cfgs = cfg.parse_args(parser)

    # ------------
    # seed
    # ------------
    pl.utilities.seed.seed_everything(cfgs.seed)

    # ------------
    # data
    # ------------
    dm = DataModule(cfgs)

    # ------------
    # model
    # ------------
    model = LitProject(cfgs)
    if cfgs.load_from_checkpoint is not None:
        # Load checkpoint
        ckpt = torch.load(cfgs.load_from_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt["state_dict"], strict=False
        )
        print(
            f"[ckpt] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}."
        )
        print(f"[ckpt] Load checkpoint from {cfgs.load_from_checkpoint}.")

    # ------------
    # trainer
    # ------------
    logger = pl.loggers.TestTubeLogger("logs", name=cfgs.name, create_git_tag=True)
    trainer = pl.Trainer.from_argparse_args(
        cfgs,
        logger=logger,
        callbacks=callbacks(cfgs),
        # === Training Setting ===
        gpus=-1,
        accelerator="ddp",
        sync_batchnorm=True,
        plugins=DDPPlugin(find_unused_parameters=False, sync_batchnorm=True),
        # === Debug Setting ===
        # profiler="simple",
        # overfit_batches=10,
        # track_grad_norm=2,
    )

    # ------------
    # training
    # ------------
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
