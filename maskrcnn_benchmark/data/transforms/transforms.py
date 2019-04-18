# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)

        image, padding = self.pad(image, size)
        target = target.pad(padding, image.size)

        '''
        import cv2,numpy
        im_tmp = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        for i in range(target.bbox.shape[0]):
            bbox = target.bbox[i]
            im_tmp = cv2.rectangle(im_tmp,(bbox[0],bbox[1]),(bbox[2], bbox[3]),(255,0,0),2)
        cv2.imshow("res",im_tmp)
        print(image.size)
        cv2.waitKey(0)
        '''
        return image, target

    def pad(self, image, size):
        
        l_pad = (self.max_size - size[1]) // 2
        r_pad = (self.max_size - size[1] + 1) // 2
        t_pad = (self.max_size - size[0]) // 2
        b_pad = (self.max_size - size[0] + 1) // 2

        image = F.pad(image, (l_pad, t_pad, r_pad, b_pad))
        return image , (l_pad, r_pad, t_pad, b_pad)

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
