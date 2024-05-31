# Copyright (C) 2022 Leonardo Romor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Simple Tiny ImageNet dataset utility class for pytorch."""
import argparse
import os

import shutil

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset, DataLoader

from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, DATA_DIRECTORY
from oodd.datasets import transforms
from oodd.datasets import BaseDataset


def normalize_tin_val_folder_structure(path,
                                       images_folder='images',
                                       annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
            and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('Validation folder is empty.')
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


class TinyImageNetFolder(ImageFolder):
    """Dataset for TinyImageNet-200"""
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root, train=True, download=False, **kwargs):
        if train:
            split = 'train'
        else:
            split = 'val'

        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self):
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url, self.data_root, filename=self.filename,
            remove_finished=True, md5=self.zip_md5)
        assert 'val' in self.splits
        normalize_tin_val_folder_structure(
            os.path.join(self.dataset_folder, 'val'))


class TinyImageNet(BaseDataset):
    _data_source = TinyImageNetFolder
    _split_args = {TRAIN_SPLIT: {"train": True}, 'val': {"train": False}}

    default_transform = None

    def __init__(
            self,
            split=TRAIN_SPLIT,
            root=DATA_DIRECTORY,
            transform=None,
            target_transform=None,
    ):
        super().__init__()

        transform = self.default_transform if transform is None else transform
        self.dataset = self._data_source(
            **self._split_args[split], root=root, transform=transform,
            target_transform=target_transform, download=True
        )

    @classmethod
    def get_argparser(cls):
        parser = argparse.ArgumentParser(description=cls.__name__)
        parser.add_argument("--root", type=str, default=DATA_DIRECTORY,
                            help="Data storage location")
        return parser

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TinyImageNetDequantized(TinyImageNet):

    # rewrite the __getitem__ function to return the dequantized image
    def __getitem__(self, idx):
        default_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
                transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
                transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
            ]
        )
        x, y = self.dataset[idx]
        x = default_transform(x)
        return x, y
