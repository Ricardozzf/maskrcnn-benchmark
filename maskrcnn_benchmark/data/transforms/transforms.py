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

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

class RandomCrop(object):

    def __init__(self, iou_thresh=0.5):
        # scale realtive to the smaller size of the image
        self.iou_thresh = iou_thresh

    def get_cropSize(self, image_size, dstSize=1200):
        w, h = image_size
        cropSize = min(min(w,h),dstSize/4)
        return cropSize

    def get_safe_params(self, image_size, crop_size, target):
        w, h = image_size
        target = target.convert("xyxy")
        bboxes = target.bbox
        if target.has_field("ignore"):
            ig = 1 - target.extra_fields["ignore"].squeeze(1)
            bboxes = target.bbox[ig.nonzero()]
        if bboxes.shape[0] == 0:
            return 0, 0, min(w,h)
        

        values_min, _ = bboxes.min(0)
        values_max, _ = bboxes.max(0)
        if values_min.ndimension() == 1:
            values_min = values_min.unsqueeze(0)
        if values_max.ndimension() == 1:
            values_max = values_max.unsqueeze(0)
        xmin = values_min[0,0].item()
        ymin = values_min[0,1].item()
        xmax = values_max[0,2].item()
        ymax = values_max[0,3].item()

        if xmax - xmin + 1 > crop_size or ymax - ymin +1 > crop_size:
            targets_num = bboxes.shape[0]
            pos = random.randint(0, targets_num - 1)
            cx1, cy1, cx2, cy2 = bboxes[pos]
            cx = (cx1 + cx2).item() / 2
            cy = (cy1 + cy2).item() / 2
            
            x1 = max(0, cx - crop_size / 2)
            y1 = max(0, cy - crop_size / 2)

            x1 = min(w - crop_size, x1)
            y1 = min(h- crop_size, y1)

            return x1, y1, crop_size
        
        x1 = random.randint(max(int(xmax - crop_size), 0), min(int(xmin), w - crop_size))
        y1 = random.randint(max(int(ymax - crop_size), 0), min(int(ymin), h - crop_size))
        return x1, y1, crop_size

    def __call__(self, image, target):
        image_size = image.size
        crop_size = self.get_cropSize(image_size)

        original_target = target.copy_with_fields(list(target.extra_fields.keys()))
 
        # guard against no box available after crop
        x1, y1, crop_size = self.get_safe_params(image_size, crop_size, original_target)
        box = (x1, y1, x1+crop_size-1, y1+crop_size-1)
        target = original_target.crop(box) # re-crop original target
        ious = target.area() / original_target.area()
        indices = ious >= self.iou_thresh
        
        target.bbox = target.bbox[indices]
        if target.has_field("ignore"):
            target.extra_fields["ignore"] = target.extra_fields["ignore"][indices]
        image = F.crop(image, y1, x1, crop_size, crop_size)
        return image, target
