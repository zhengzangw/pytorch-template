import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from ..augmentation import get_aug


def get_dst(name, split="train"):
    assert split in ["train", "val", "test"]

    if name == "mnist":
        mnist_transform = get_aug(name)
        if split == "train":
            return torchvision.datasets.MNIST(
                "./data/mnist", train=True, transform=mnist_transform, download=True
            )
        else:
            return torchvision.datasets.MNIST(
                "./data/mnist", train=False, transform=mnist_transform, download=True
            )
    elif name == "cifar10":
        transform_train = get_aug("cifar10_train")
        transform_test = get_aug("cifar10_test")
        if split == "train":
            return torchvision.datasets.CIFAR10(
                root="./data/cifar10",
                train=True,
                download=True,
                transform=transform_train,
            )
        else:
            return torchvision.datasets.CIFAR10(
                root="./data/cifar10",
                train=False,
                download=True,
                transform=transform_test,
            )
    else:
        raise NotImplementedError


class DataModule(pl.LightningDataModule):
    def __init__(self, cfgs, pin_memory=True):
        super().__init__()
        self.cfgs = cfgs
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dst = get_dst(self.cfgs.dataset, split="train")
            self.val_dst = get_dst(self.cfgs.dataset, split="val")
        elif stage == "test" or stage is None:
            self.test_dst = get_dst(self.cfgs.dataset, split="test")

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

