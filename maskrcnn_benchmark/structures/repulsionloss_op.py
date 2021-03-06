import numpy as np
import torch
import torch.nn as nn
import random


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def onehot_iou(a ,b):
    xx1 = torch.max(a[:, 0], b[:, 0])
    yy1 = torch.max(a[:, 1], b[:, 1])
    xx2 = torch.min(a[:, 2], b[:, 2])
    yy2 = torch.min(a[:, 3], b[:, 3])
    iw = torch.clamp(xx2 - xx1 + 1, min=0)
    ih = torch.clamp(yy2 - yy1 + 1, min=0)

    area1 = (a[: ,2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area2 = (b[: ,2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    return iw * ih / (area1 + area2)

def IoG(box_a, box_b):

    inter_xmin = torch.max(box_a[:, 0], box_b[:, 0])
    inter_ymin = torch.max(box_a[:, 1], box_b[:, 1])
    inter_xmax = torch.min(box_a[:, 2], box_b[:, 2])
    inter_ymax = torch.min(box_a[:, 3], box_b[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    return I / G


def smooth_ln(x, smooth):
    return torch.where(
        torch.le(x, smooth),
        -torch.log(1 - x),
        ((x - smooth) / (1 - smooth)) - np.log(1 - smooth)
    )

def smooth_l1(input, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


