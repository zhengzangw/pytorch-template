import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from .models import get_model


class LitProject(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters(cfgs)
        self.cfgs = cfgs

        # model
        self.model = get_model(cfgs.model)(**cfgs.model_args)
        self.criterion = get_model(cfgs.criterion)()

        # metrics
        metrics = torchmetrics.Accuracy()
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x):
        return self.model(x)

    # ------------
    # train
    # ------------

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)

        # metric
        pred = logits.argmax(dim=1)
        self.train_metrics(pred, labels)
        self.log("train/loss", loss)

        return loss

    def training_epoch_end(self, outputs):
        # a bug in pl-1.2: metric not auto reset
        self.log(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    # ------------
    # validation
    # ------------

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        # loss
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        self.log("val/loss", loss)

        # metric
        pred = logits.argmax(dim=1)
        self.val_metrics(pred, labels)

    def validation_epoch_end(self, outputs):
        # a bug in pl-1.2: metric not auto reset
        self.log(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    # ------------
    # test
    # ------------

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    # ------------
    # optim
    # ------------

    def configure_optimizers(self):
        params = self.model.parameters()
        # ADAM
        if self.cfgs.optimizer == "Adam":
            optimizer = optim.Adam(params, lr=self.cfgs.lr)
        # ADAMW
        elif self.cfgs.optimizer == "AdamW":
            optimizer = optim.AdamW(params, lr=self.cfgs.lr)
        # SGD
        elif self.cfgs.optimizer == "SGD":
            optimizer = optim.SGD(
                params, lr=self.cfgs.lr, momentum=0.9, weight_decay=5e-4,
            )
        else:
            raise NotImplementedError

        # Sched
        schs = []

        return [optimizer], schs

