import torch
import torchvision
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, Dataset

class LoaderGenerator:
    def __init__(self, root, dataset_name, train_batch_size=1, test_batch_size=1, num_workers=0, kwargs={}):
        self.root = root
        self.dataset_name = str.lower(dataset_name)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.items = []
        self._train_set = None
        self._test_set = None
        self._calib_set = None
        self.train_transform = None
        self.test_transform = None
        self.train_loader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': kwargs.get('pin_memory', True),
            'drop_last': kwargs.get('drop_last', False)
        }
        self.test_loader_kwargs = self.train_loader_kwargs.copy()
        self.load()

    def load(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = datasets.ImageFolder(os.path.join(self.root, 'train'), self.train_transform)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = datasets.ImageFolder(os.path.join(self.root, 'val'), self.test_transform)
        return self._test_set

    @property
    def val_set(self):
        return self.test_set

    def train_loader(self):
        assert self.train_set is not None
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, **self.train_loader_kwargs)

    def test_loader(self):
        assert self.test_set is not None
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, **self.test_loader_kwargs)

    def val_loader(self):
        assert self.val_set is not None
        return DataLoader(self.val_set, batch_size=self.test_batch_size, shuffle=False, **self.test_loader_kwargs)

class ImageNetLoaderGenerator(LoaderGenerator):
    def load(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = datasets.ImageFolder(os.path.join(self.root, 'train'), self.train_transform)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = datasets.ImageFolder(os.path.join(self.root, 'val'), self.test_transform)
        return self._test_set