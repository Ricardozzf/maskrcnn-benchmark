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
    gaussian_kernel = torch.squeeze(gaussian_kernel)
    
    return gaussian_kernel

def get_gaussian_target(targets, imageSize, maskResize):
    """
    Arguments:
        targets (list[BoxList])

    Returns:
        gaussian_targets (list[Tensor])
    """
    gaussian_targets = []
    for target in targets:

        bboxes = target.bbox
        mode = target.mode
        feature_target = torch.zeros(imageSize)
        if mode not in ("xyxy"):
            raise ValueError("mode should be 'xyxy'")
        for bbox in bboxes:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            width = x2 - x1
            height = y2 - y1
            
            gaussian_feature = get_gaussian_kernel(height, width)
            
            feature_target[y1:y2, x1:x2] = feature_target[y1:y2, x1:x2] + gaussian_feature

        feature_target = Ft.resize(feature_target, maskResize)

        gaussian_targets.append(feature_target)
    
    return gaussian_targets
