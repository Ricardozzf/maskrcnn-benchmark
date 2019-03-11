import math
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as Ft

import cv2

def get_gaussian_kernel(rectX=51, rectY=51, sigma=10, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    kernel_size = 51
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    
    mean = (kernel_size - 1)/2.

    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    xy_grid = xy_grid - mean
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    #gaussian_kernel = (1./(2.*math.pi*variance)) *\
    gaussian_kernel = 1 *\
                      torch.exp(
                          -torch.sum((xy_grid)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    #gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
    
    gaussian_kernel = torch.unsqueeze(gaussian_kernel, 0)
    gaussian_kernel = torch.unsqueeze(gaussian_kernel, 0)
    gaussian_kernel = F.interpolate(gaussian_kernel, size=[rectX, rectY],mode='bilinear',align_corners=True)
    gaussian_kernel = torch.squeeze(gaussian_kernel, dim=1)
    
    return gaussian_kernel

def get_gaussian_target(targets, imageSize, maskResize = None):
    """
    Arguments:
        targets (list[BoxList])

    Returns:
        gaussian_targets (list[Tensor])
    """
    gaussian_targets = []
    mseloss_weights = []
    for target in targets:

        bboxes = target.bbox
        mode = target.mode
        feature_target = torch.zeros(1,imageSize[1],imageSize[2],requires_grad=False)

        if mode not in ("xyxy"):
            raise ValueError("mode should be 'xyxy'")
        for bbox in bboxes:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            if x1 < 0:
                x1 = 0
                x2 = x2 - x1
            if y1 < 0:
                y1 = 0
                y2 = y2 - y1
            if x2 > imageSize[2]:
                x2 = imageSize[2]
                x1 = x1 - (x2 - imageSize[2])
            if  y2 > imageSize[1]:
                y2 = imageSize[1]
                y1 = y1 - (y2 - imageSize[1])

            width = x2 - x1
            height = y2 - y1
            if width < 5 or height < 5:
                continue
            
            gaussian_feature = get_gaussian_kernel(height, width)
            
            try:
                feature_target[0, y1:y2, x1:x2] = feature_target[0, y1:y2, x1:x2] + gaussian_feature
            except :
                import pdb; pdb.set_trace()

        #feature_target = Ft.resize(feature_target, maskResize)
        feature_target = feature_target.unsqueeze(0)
        feature_target = F.interpolate(feature_target, size=[maskResize[1], maskResize[0]],mode='bilinear',align_corners=True)
        mseloss_weight = torch.exp(torch.pow(feature_target.squeeze(),5)).view(1, -1)
        mseloss_weight = F.softmax(mseloss_weight).view(1, 1, maskResize[1], maskResize[0])
        mseloss_weight[mseloss_weight < 0.1] = 0.1

        #feature_target = feature_target / feature_target.max()
        gaussian_targets.append(feature_target)
        mseloss_weights.append(mseloss_weight)
    return gaussian_targets, mseloss_weights

def weighted_mse_loss(input, target, weights):
    #import pdb; pdb.set_trace()
    #out = (torch.log(input+1)-target)**2
    #out = out * weights.expand_as(out)
    #loss = out.mean() # or sum over whatever dimensions
    import pdb; pdb.set_trace()
    Lsa = smooth_l1(input, target, 1).mean()
    Lsc = 1 - (input * target).sum() / ((input **2).sum() + (target **2).sum())
    #import pdb; pdb.set_trace()
    #loss = smoothLoss * weights
    #loss = smoothLoss.mean()
    return Lsa + Lsc

def smooth_l1(input, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss
