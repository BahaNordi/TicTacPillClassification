# Taken and adapted by Ali Ehteshami Bejnordi from the github repo of zijundeng
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
import numbers
import random
import numpy as np

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img):
        img = Image.fromarray(img, mode='L')

        for aug in self.augmentations:
            img = aug(img)
        return np.array(img, dtype=np.uint8)


class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        if random.random() < 0.5:
            rotate_degree = random.choice(self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))
