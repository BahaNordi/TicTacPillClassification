import os
import numpy as np
# import scipy.misc as misc
import torch
from torch.utils import data
from train_utils.train_config import config
from PIL import Image



class TicTacLoader(data.Dataset):

    def __init__(self, root_dir, tictac_file_list, augmentations=None, transform_image=None):
        self.augmentations = augmentations
        self.transform_image = transform_image
        self.mean = config.mean_val
        if root_dir:
            self.root = root_dir
        else:
            self.root = os.getcwd()

        self.file_list_dir = tictac_file_list
        # line.rstrip remove \n enter from the end of file
        self.ids = [os.path.join(self.root, line.rstrip()) for line in open(self.file_list_dir)]

    def __len__(self):
        """__len__"""
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.ids[index]
        label = 0
        if "defect" in img_path:
            label = 1
        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file.".format(img_path))

        if self.augmentations is not None:
            img = self.augmentations(img)

        if self.transform_image:
            img = self.transform(img)

        img = torch.from_numpy(img).float()
        return img, label

    def transform(self, img):
        img = img.astype(float)
        img -= self.mean
        img /= 255.0
        img = img[None, :, :]
        return img


if __name__ == "__main__":
    root_dir1 = '/home/baha/codes/tictac/data/'
    tictac_file_list1 = '/home/baha/codes/tictac/data/list_of_files.txt'
    my_loader = TicTacLoader(root_dir1, tictac_file_list1)
