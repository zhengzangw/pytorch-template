import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    has_teardown_None = False  # a bug on pl-1.2

    def __init__(self, cfgs, pin_memory=True):
        super().__init__()
        self.cfgs = cfgs
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dst = None
            self.val_dst = None
        elif stage == "test" or stage is None:
            self.test_dst = None

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dst,
            batch_size=self.cfgs.batch_size,
            num_workers=self.cfgs.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dst,
            batch_size=self.cfgs.batch_size,
            num_workers=self.cfgs.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dst,
            batch_size=self.cfgs.batch_size,
            num_workers=self.cfgs.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_loader


if __name__ == "__main__":
    from project.cfg import parse_args

    cfgs = parse_args()

    dm = DataModule(cfgs)
    dm.prepare_data()
    dm.setup(stage="fit")

    for batch in dm.train_dataloader():
        breakpoint()
        break
    for batch in dm.val_dataloader():
        breakpoint()
        break
    for batch in dm.test_dataloader():
        break

