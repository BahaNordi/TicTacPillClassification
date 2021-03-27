import torch
import torch.utils.data as torch_data

from data_loader.augmentation import *
from data_loader.tictac_loader import TicTacLoader


class TicTacDataLoader(object):
    def __init__(self, config, root_dir=None, tictac_train_list=None, tictac_val_list=None, tictac_test_list=None):
        self.root_dir = root_dir
        self.tictac_train_list = tictac_train_list
        self.tictac_val_list = tictac_val_list
        self.tictac_test_list = tictac_test_list
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.samples_weight = torch.from_numpy(config.samples_weight).double()
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @property
    def train_loader(self):
        if not self._train_loader:
            augment = Compose([RandomHorizontalFlip(), RandomRotate([90, 180, 270]), RandomCrop((120, 120))])
            self.train_set = TicTacLoader(self.root_dir, self.tictac_train_list,
                                          augmentations=augment, transform_image=True)

            sampler = torch.utils.data.sampler.WeightedRandomSampler(self.samples_weight, len(self.samples_weight))
            self._train_loader = torch_data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                       num_workers=self.num_workers, sampler=sampler)
        return self._train_loader

    @property
    def val_loader(self):
        if not self._val_loader:
            augment = Compose([RandomCrop((120, 120))])
            self.val_set = TicTacLoader(self.root_dir, self.tictac_val_list,
                                        augmentations=augment, transform_image=True)
            self._val_loader = torch_data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                                     num_workers=self.num_workers)
        return self._val_loader

    @property
    def test_loader(self):
        if not self._val_loader:
            augment = Compose([RandomHorizontalFlip(), RandomRotate([90, 180, 270]), RandomCrop((120, 120))])
            self.val_set = TicTacLoader(self.root_dir, self.tictac_test_list,
                                        augmentations=augment, transform_image=True)
            self._val_loader = torch_data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                                     num_workers=self.num_workers)
        return self._val_loader
