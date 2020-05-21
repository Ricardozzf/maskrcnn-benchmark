# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        if reference_boxes.shape[1] != 8:
            raise RuntimeError("gt bbox should be N×6 dims!")

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        gt_v_w = reference_boxes[:, 6]
        gt_v_h = reference_boxes[:, 7]
        gt_v_cx = reference_boxes[:, 4] + 0.5 * gt_v_w
        gt_v_cy = reference_boxes[:, 5] + 0.5 * gt_v_h

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_vx = wx * (gt_v_cx - ex_ctr_x) / ex_widths
        targets_vy = wy * (gt_v_cy - ex_ctr_y) / ex_heights
        targets_vw = ww * torch.log(gt_v_w / ex_widths)
        targets_vh = wh * torch.log(gt_v_h / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, \
            targets_vx, targets_vy, targets_vw, targets_vh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::8] / wx
        dy = rel_codes[:, 1::8] / wy
        dw = rel_codes[:, 2::8] / ww
        dh = rel_codes[:, 3::8] / wh
        vx = rel_codes[:, 4::8] / wx
        vy = rel_codes[:, 5::8] / wy
        vw = rel_codes[:, 6::8] / ww
        vh = rel_codes[:, 7::8] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        vw = torch.clamp(vw, max=self.bbox_xform_clip)
        vh = torch.clamp(vh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_v_x = vx * widths[:, None] + ctr_x[:, None]
        pred_v_y = vy * heights[:, None] + ctr_y[:, None]
        pred_vw = torch.exp(vw) * widths[:, None]
        pred_vh = torch.exp(vh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::8] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::8] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::8] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::8] = pred_ctr_y + 0.5 * pred_h - 1
        # vx
        pred_boxes[:, 4::8] = pred_v_x - 0.5 * pred_vw
        # vy
        pred_boxes[:, 5::8] = pred_v_y - 0.5 * pred_vh
        # vw
        pred_boxes[:, 6::8] = pred_vw
        # vh
        pred_boxes[:, 7::8] = pred_vh

        return pred_boxes
