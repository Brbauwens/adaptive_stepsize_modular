import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import random
import numpy as np
from pathlib import Path

from datasets import load_from_disk
from tools.load_imagenet_parquet import load_ImageNet100


debug = 0
paths = ["/opt/software/datasets/cifar/", "../data"]
#paths = ["../data"]

def load_data(dataset, manual_seed = -1, train_transform=None, test_transform=None):

    data2func = {
            "FashionMNIST"  : load_FashionMNIST, 
            "MNIST"         : load_MNIST, 
            "CIFAR10"       : load_CIFAR10, 
            "CIFAR100"      : load_CIFAR100, 
            "PointsDataset" : load_pointsDataset,
            "SVHN"          : load_SVHN,
            "ImageNet100"   : load_ImageNet100
            }
    if dataset in data2func:
        return data2func[dataset](manual_seed)
    assert type(dataset) is not str, f"I don't know {dataset}. I do know {', '.join(data2func.keys())}"

    if train_transform is None:
        train_transform = ToTensor()
    if test_transform is None:
        test_transform = train_transform
    return _load_data(dataset, manual_seed, train_transform, test_transform)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _load_data(dataset, manual_seed, train_transform=None, test_transform=None, debug=False):

    directory = "../data"
    for directory in paths:
        try :
            train_dataset = dataset(
                root=directory, 
                train=True,
                download=True,
                transform=train_transform
            )

            test_dataset = dataset(
                root=directory, 
                train=False,
                download=True,
                transform=test_transform
            )

            break
        except:
            pass

    if manual_seed < 0:
        if debug or 'debug' in globals() and debug >= 1:
            DlClass = LimitedDataLoader # Defined below
            train_kwargs = {'max_batches' : debug, 'batch_size' : 100, 'shuffle' : False} 
            test_kwargs  = {'max_batches' : 1,     'batch_size' : 100, 'shuffle' : False} 
            print("debug mode      ", end='')
        else :
            DlClass = DataLoader
            train_kwargs, test_kwargs = {'batch_size' : 128, 'shuffle' : True}, {'batch_size' : 1024} 

        train_dl = DlClass(train_dataset, num_workers=2, pin_memory=True, **train_kwargs)
        test_dl = DlClass(test_dataset, num_workers=2, pin_memory=True, **test_kwargs)

        print(f"training size: {len(train_dataset)} -> {train_dl.batch_size} x {len(train_dl)}      "
            f"validation size: {len(test_dataset)} -> {test_dl.batch_size} x {len(test_dl)}")

        return train_dl, test_dl
    else:
        gen = torch.Generator()
        gen.manual_seed(manual_seed)

        train_dl = DataLoader(train_dataset, batch_size=128, shuffle = True, num_workers=2, pin_memory=True,\
                              worker_init_fn=seed_worker, generator=gen)
        test_dl = DataLoader(test_dataset, batch_size=1024, shuffle = False, num_workers=2, pin_memory=True,\
                             worker_init_fn=seed_worker, generator=gen)

        print(f"training size: {len(train_dataset)} -> {train_dl.batch_size} x {len(train_dl)}      "
            f"validation size: {len(test_dataset)} -> {test_dl.batch_size} x {len(test_dl)}    "
            f"manual_seed: {manual_seed}"
            )

        return train_dl, test_dl

# transform is not used ?????
def load_MNIST():
    #transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    #])
    return load_data(datasets.MNIST)


def load_FashionMNIST():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return load_data(datasets.FashionMNIST, train_transform=train_transform, test_transform=test_transform)


stats_cifar = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats_cifar,inplace=True)
])

test_transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats_cifar)])

def load_CIFAR10(manual_seed):
    return _load_data(datasets.CIFAR10, manual_seed, train_transform=train_transform_cifar, test_transform=test_transform_cifar)

def load_CIFAR100(manual_seed):
    return _load_data(datasets.CIFAR100, manual_seed, train_transform=train_transform_cifar, test_transform=test_transform_cifar)

stats_svhn = ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))

train_transform_svhn = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats_svhn,inplace=True)
])

test_transform_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats_svhn)])

def load_SVHN(manual_seed):
    DATASET_PATH = "/opt/software/datasets/svhn/"
    BATCH_SIZE = 128
    train_dataset = datasets.SVHN(root=DATASET_PATH, split='train', download=True, transform=train_transform_svhn)
    test_dataset = datasets.SVHN(root=DATASET_PATH, split='test', download=True, transform=test_transform_svhn)

    train_dataloader = None
    if manual_seed < 0:
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    else:
        gen = torch.Generator()
        gen.manual_seed(manual_seed)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, num_workers=2, pin_memory=True,\
                                worker_init_fn=seed_worker, generator=gen)

    test_dataloader = DataLoader(test_dataset, batch_size=1024, num_workers=2, pin_memory=True)
    print(f"training size: {len(train_dataset)} -> {train_dataloader.batch_size} x {len(train_dataloader)}      "
        f"validation size: {len(test_dataset)} -> {test_dataloader.batch_size} x {len(test_dataloader)}")
    return train_dataloader, test_dataloader

class LimitedDataLoader(DataLoader):
    """Class that reduces the size of the training set for debugging.""" 
    def __init__(self, *args, max_batches=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate((iterator := super().__iter__())):
            if self.max_batches and i >= self.max_batches:
                break
            yield batch
        # chatgpt says that cleans up workers by itself, but with large datasets it sometimes fails. 
        if hasattr(iterator, "_shutdown_workers"): 
            iterator._shutdown_workers()

    def __len__(self):
        return min(super().__len__(), self.max_batches)

class PointsDataset(Dataset):
    """
    A simple dataset for debugging purposes. 
    The inputs are standard normal distributed in at least 2 dimensions.  The labels are the sign of the sum of the 2 first candidates.
    One can vary the difficulty by changing the number of dimensions. 
    For more stable comparison between experiments, the datasets are cached in a hidden subdirectory .dataset/pointsDataset/
    """
    _cache_dir = Path(".data/pointsDataset/")

    def __init__(self, dim, n, train_or_test):
        self._dim, self._n, self._train_or_test = dim, n, train_or_test

        if not self._filename().exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            X = torch.randn(n, dim)
            y = (X[:, 0] + X[:, 1] > 0).long()
            data = {"X" : X, "y" : y}
            torch.save(data, self._filename())
        else :
            data = torch.load(self._filename())

        self.X = data["X"]
        self.y = data["y"]

        assert len(self.X) == len(self.y), "Feature/label size mismatch"

        self.classes = [0, 1]
        self.num_classes = 2


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _filename(self):
        return self._cache_dir / f"dim{self._dim}_{self._train_or_test}{self._n}.pt"


def load_pointsDataset(dim=20, n_train=2000, n_test=1000):

    train_loader = DataLoader(
        PointsDataset(dim, n_train, 'train'), 
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        PointsDataset(dim, n_test, 'test'), 
        batch_size=100,
        shuffle=False,  
        num_workers=0
    )

    return train_loader, test_loader

def get_device():
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
