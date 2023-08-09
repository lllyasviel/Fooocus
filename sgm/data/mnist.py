import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTDataDictWrapper(Dataset):
    def __init__(self, dset):
        super().__init__()
        self.dset = dset

    def __getitem__(self, i):
        x, y = self.dset[i]
        return {"jpg": x, "cls": y}

    def __len__(self):
        return len(self.dset)


class MNISTLoader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, prefetch_factor=2, shuffle=True):
        super().__init__()

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else 0
        self.shuffle = shuffle
        self.train_dataset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/", train=True, download=True, transform=transform
            )
        )
        self.test_dataset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/", train=False, download=True, transform=transform
            )
        )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )


if __name__ == "__main__":
    dset = MNISTDataDictWrapper(
        torchvision.datasets.MNIST(
            root=".data/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
            ),
        )
    )
    ex = dset[0]
