import random
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor



debug = 0
#paths = ["/opt/software/datasets/cifar/", "../data"]
paths = ["../data"]



def _make_generator(seed, stream=0):
    """Create an independent RNG stream for one DataLoader."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")

    generator = torch.Generator()
    generator.manual_seed((seed + stream) % 2**64)
    return generator


def _seed_worker(worker_id):
    """Seed non-PyTorch RNGs inside a DataLoader worker.

    DataLoader has already seeded PyTorch for this worker.  Deriving the
    NumPy and Python seeds from that value keeps custom transforms that use
    either library reproducible as well.
    """
    del worker_id  # The worker-specific value is already in torch.initial_seed().
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(dataset, train_transform=None, test_transform=None, seed=None):
    """Load a dataset, optionally using a reproducible random seed.

    Repeated calls with the same ``seed`` produce the same training shuffle,
    worker seeds, and random torchvision transforms (for example RandomCrop).
    Separate RNG streams are used for training and test loaders, so their
    results do not depend on which loader is iterated first.

    Successive epochs from one DataLoader can still differ, as is customary
    for training, but the complete epoch sequence is reproducible when the
    loader is reconstructed with the same seed.
    """
    data2func = {
            "FashionMNIST"  : load_FashionMNIST,
            "MNIST"         : load_MNIST,
            "CIFAR10"       : load_CIFAR10,
            "CIFAR100"      : load_CIFAR100,
            "PointsDataset" : load_pointsDataset
            }
    if dataset in data2func:
        return data2func[dataset](seed=seed)
    assert type(dataset) is not str, f"I don't know {dataset}. I do know {', '.join(data2func.keys())}"

    if train_transform is None:
        train_transform = ToTensor()
    if test_transform is None:
        test_transform = train_transform
    return _load_data(dataset, train_transform, test_transform, seed=seed)


def _load_data(dataset, train_transform=None, test_transform=None, debug=False, seed=None):

    directory = "../data"
    for directory in paths:
        try:
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
        except Exception:
            pass

    if debug or 'debug' in globals() and debug >= 1:
        DlClass = LimitedDataLoader  # Defined below
        train_kwargs = {'max_batches': debug, 'batch_size': 100, 'shuffle': False}
        test_kwargs = {'max_batches': 1, 'batch_size': 100, 'shuffle': False}
        print("debug mode      ", end='')
    else:
        DlClass = DataLoader
        train_kwargs, test_kwargs = {'batch_size': 128, 'shuffle': True}, {'batch_size': 1024}

    # Keep the train and test RNG streams independent.  The generator controls
    # both RandomSampler (shuffle=True) and the initial PyTorch seed of workers.
    train_dl = DlClass(
        train_dataset,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=_make_generator(seed, stream=0),
        **train_kwargs,
    )
    test_dl = DlClass(
        test_dataset,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=_make_generator(seed, stream=1),
        **test_kwargs,
    )

    print(f"training size: {len(train_dataset)} -> {train_dl.batch_size} x {len(train_dl)}      "
          f"validation size: {len(test_dataset)} -> {test_dl.batch_size} x {len(test_dl)}")

    return train_dl, test_dl


# transform is not used ?????
def load_MNIST(seed=None):
    #transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    #])
    return load_data(datasets.MNIST, seed=seed)


def load_FashionMNIST(seed=None):
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

    return load_data(
        datasets.FashionMNIST,
        train_transform=train_transform,
        test_transform=test_transform,
        seed=seed,
    )


stats_cifar = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats_cifar, inplace=True)
])

test_transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats_cifar)])


def load_CIFAR10(seed=None):
    return _load_data(
        datasets.CIFAR10,
        train_transform=train_transform_cifar,
        test_transform=test_transform_cifar,
        seed=seed,
    )


def load_CIFAR100(seed=None):
    return _load_data(
        datasets.CIFAR100,
        train_transform=train_transform_cifar,
        test_transform=test_transform_cifar,
        seed=seed,
    )


class LimitedDataLoader(DataLoader):
    """Class that reduces the size of the training set for debugging."""
    def __init__(self, *args, max_batches=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_batches = max_batches

    def __iter__(self):
        iterator = super().__iter__()
        try:
            for i, batch in enumerate(iterator):
                if self.max_batches and i >= self.max_batches:
                    break
                yield batch
        finally:
            # DataLoader normally cleans up workers itself.  Keep the explicit
            # shutdown used by the original debugging loader as a safeguard.
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
            data = {"X": X, "y": y}
            torch.save(data, self._filename())
        else:
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


def load_pointsDataset(dim=20, n_train=2000, n_test=1000, seed=None):

    train_loader = DataLoader(
        PointsDataset(dim, n_train, 'train'),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=_make_generator(seed, stream=0),
    )

    test_loader = DataLoader(
        PointsDataset(dim, n_test, 'test'),
        batch_size=100,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=_make_generator(seed, stream=1),
    )

    return train_loader, test_loader
