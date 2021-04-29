import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def callbacks(args, monitor="val/accuracy"):
    callbacks_list = []

    # log learning rate
    callbacks_list.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

    # best model checkpoint
    callbacks_list.append(
        pl.callbacks.ModelCheckpoint(
            monitor=monitor,
            mode="max",
            save_top_k=1,
            save_last=True,
            # filename
            # new style will be supported in pl-1.3
            filename="{epoch:02d}",
            # auto_insert_metric_name=False,
            # filename="epoch{epoch:02d}-val_iou{val/IoU:.2f}",
        )
    )

    # early stop
    if "patience" in args:
        callbacks_list.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor=monitor, patience=args.patience, mode="max"
            )
        )

    return callbacks_list
