# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from maskrcnn_benchmark.structures.repulsionloss_op import IoG
from maskrcnn_benchmark.structures.repulsionloss_op import smooth_ln
from maskrcnn_benchmark.structures.repulsionloss_op import smooth_l1
from maskrcnn_benchmark.structures.repulsionloss_op import calc_iou
from maskrcnn_benchmark.structures.repulsionloss_op import onehot_iou
from maskrcnn_benchmark.modeling import registry

@registry.RPN_LOSS.register("RPNLossComputation")
class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level in zip(
            objectness, box_regression
        ):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
        
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())
        
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = registry.RPN_LOSS[cfg.MODEL.RPN.RPN_LOSS](matcher, fg_bg_sampler, box_coder)
    return loss_evaluator

#*****************************************************************#
#                     add repulsion loss                          #
#                     author: zhanfan zou                         #
#                     data: 2018.11.19                            #
#*****************************************************************#
import numpy as np
import random

@registry.RPN_LOSS.register("RPNRepLossComputation")
class RPNRepLossComputation(object):
    """
    This class computes the Repulsion loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )
            
            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        
        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level in zip(
            objectness, box_regression
        ):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)

        # keep bbox_regression
        box_regression_reploss = cat(box_regression_flattened, dim=1)
        batches = box_regression_reploss.shape[0]
        num_anchors = box_regression_reploss.shape[1]

        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        #import pdb
        #pdb.set_trace()
        box_loss_tmp = smooth_l1(box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,)


        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )     


        ######################################################
        anchor_flattened = [] 
        for anchor_per in anchors:
            anchor_flattened.append(anchor_per.bbox)
        #assert len(anchor_flattened) <2,"Multi level anchor!"
        #anchors_bbox = cat(anchor_flattened, dim=0).reshape(-1,4)
        anchors_bbox = anchor_flattened

        targets_bbox_flattened = []
        for targets_bbox_per in targets:
            targets_bbox_flattened.append(targets_bbox_per.bbox)
        #import pdb; pdb.set_trace()
        #targets_box = cat(targets_bbox_flattened, dim=0).reshape(-1,4)
        targets_box = targets_bbox_flattened

        RepGT_losses = 0
        RepBox_losses = 0
        tmp_index = 0
        for batch in range(batches):
            box_regression_dx = box_regression_reploss[batch,:,0]
            box_regression_dy = box_regression_reploss[batch,:,1]
            box_regression_dw = box_regression_reploss[batch,:,2]
            box_regression_dh = box_regression_reploss[batch,:,3]
            #assert box_regression.shape[0] == anchors_bbox.shape[0],"Invalid shape with bbox_regression && anchors!"
            
            targets_box_batch = targets_box[batch]
            anchors_bbox_batch = anchors_bbox[batch]

            inds_ge = sampled_pos_inds.ge(batch*num_anchors)
            inds_le = sampled_pos_inds.le(batch*num_anchors + num_anchors-1)
            inds_bet = inds_ge * inds_le
            sampled_pos_inds_batch = sampled_pos_inds[inds_bet] % num_anchors

            if len(sampled_pos_inds_batch) != 0:

                anchors_bbox_cx = (anchors_bbox_batch[:, 0] + anchors_bbox_batch[:, 2]) / 2.0
                anchors_bbox_cy = (anchors_bbox_batch[:, 1] + anchors_bbox_batch[:, 3]) / 2.0
                anchors_bbox_w = anchors_bbox_batch[:, 2] - anchors_bbox_batch[:, 0] + 1
                anchors_bbox_h = anchors_bbox_batch[:, 3] - anchors_bbox_batch[:, 1] + 1
                predict_w = torch.exp(box_regression_dw) * anchors_bbox_w
                predict_h = torch.exp(box_regression_dh) * anchors_bbox_h
                predict_x = box_regression_dx * anchors_bbox_w + anchors_bbox_cx
                predict_y = box_regression_dy * anchors_bbox_h + anchors_bbox_cy

                predict_x1 = predict_x - 0.5 * predict_w
                predict_y1 = predict_y - 0.5 * predict_h
                predict_x2 = predict_x + 0.5 * predict_w
                predict_y2 = predict_y + 0.5 * predict_h

                predict_boxes = torch.stack((predict_x1, predict_y1, predict_x2, predict_y2)).t()
                predict_boxes_pos = predict_boxes[sampled_pos_inds_batch, :]
                IoU = calc_iou(anchors_bbox_batch, targets_box_batch[:, :4])  # num_anchors x num_annotations
                IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1
                
                #add RepGT losses
                IoU_pos = IoU[sampled_pos_inds_batch,:]
                IoU_max_keep, IoU_argmax_keep = torch.max(IoU_pos, dim=1, keepdim=True)  # num_anchors x 1
                for idx in range(IoU_argmax_keep.shape[0]):
                    IoU_pos[idx, IoU_argmax_keep[idx]] = -1
                IoU_sec, IoU_argsec = torch.max(IoU_pos, dim=1)
                assigned_annotations_sec = targets_box_batch[IoU_argsec, :]


                box_loss_tmp_batch = box_loss_tmp[tmp_index: tmp_index+sampled_pos_inds_batch.shape[0]]
                box_loss_tmp_batch = torch.sum(box_loss_tmp_batch, dim=1)
                IoG_to_minimize = IoG(assigned_annotations_sec, predict_boxes_pos)
                RepGT_loss = smooth_ln(IoG_to_minimize, 0.5)
                RepGT_loss = RepGT_loss * torch.lt(0.1*RepGT_loss, box_loss_tmp_batch).float()
                RepGT_loss = RepGT_loss.mean() / sampled_pos_inds.numel()
                RepGT_losses += RepGT_loss

                #add RepBox losses
                IoU_argmax_pos = IoU_argmax[sampled_pos_inds_batch].float()
                IoU_argmax_pos = IoU_argmax_pos.unsqueeze(0).t()
                predict_boxes_pos = torch.cat([predict_boxes_pos,IoU_argmax_pos],dim=1)

                predict_boxes_pos_np = predict_boxes_pos.detach().cpu().numpy()
                num_gt = targets_box_batch.shape[0]
                predict_boxes_pos_sampled = []
                box_loss_tmp_batch_sampled = []
                for id in range(num_gt):
                    index = np.where(predict_boxes_pos_np[:, 4]==id)[0]
                    if index.shape[0]:
                        idx = random.choice(range(index.shape[0]))
                        predict_boxes_pos_sampled.append(predict_boxes_pos[index[idx], :4])
                        box_loss_tmp_batch_sampled.append(box_loss_tmp_batch[index[idx]])
                predict_boxes_pos_sampled = torch.stack(predict_boxes_pos_sampled)
                box_loss_tmp_batch_sampled = torch.stack(box_loss_tmp_batch_sampled)
                iou_repbox = calc_iou(predict_boxes_pos_sampled, predict_boxes_pos_sampled)
                mask = torch.lt(iou_repbox, 1.).float()
                iou_repbox = iou_repbox * mask
                RepBox_loss = smooth_ln(iou_repbox, 0.5)
                RepBox_loss = RepBox_loss * torch.lt(0.85*RepBox_loss, box_loss_tmp_batch_sampled).float()
                RepBox_loss = RepBox_loss.sum() / sampled_pos_inds.numel()
                RepBox_losses += RepBox_loss


                tmp_index += sampled_pos_inds_batch.shape[0]
                if RepBox_losses!=RepBox_losses or RepGT_losses!=RepGT_losses or box_loss!=box_loss:
                    import pdb; pdb.set_trace()

        RepGT_losses /= batches
        RepBox_losses /= batches
        reg_loss = box_loss + 0.1 * RepGT_losses +  0.7 * RepBox_losses

           
        return objectness_loss, reg_loss


#*****************************************************************#
#                     add iou-net                                 #
#                     author: zhanfan zou                         #
#                     data: 2018.11.26                            #
#*****************************************************************#

@registry.RPN_LOSS.register("RPNIoULossComputation")
class RPNIoULossComputation(object):
    """
    This class computes the RPN-IoUNet loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level in zip(
            objectness, box_regression
        ):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

        # cat anchors dim to match regression dim
        anchor_flattened = [] 
        for anchor_per in anchors:
            anchor_flattened.append(anchor_per.bbox)
        anchors_bbox = torch.cat(anchor_flattened, dim=0)

        box_regression_dx = box_regression[:, 0]
        box_regression_dy = box_regression[:, 1]
        box_regression_dw = box_regression[:, 2]
        box_regression_dh = box_regression[:, 3]
        
        anchors_bbox_cx = (anchors_bbox[:, 0] + anchors_bbox[:, 2]) / 2.0
        anchors_bbox_cy = (anchors_bbox[:, 1] + anchors_bbox[:, 3]) / 2.0
        anchors_bbox_w = anchors_bbox[:, 2] - anchors_bbox[:, 0] + 1
        anchors_bbox_h = anchors_bbox[:, 3] - anchors_bbox[:, 1] + 1
        predict_w = torch.exp(box_regression_dw) * anchors_bbox_w
        predict_h = torch.exp(box_regression_dh) * anchors_bbox_h
        predict_x = box_regression_dx * anchors_bbox_w + anchors_bbox_cx
        predict_y = box_regression_dy * anchors_bbox_h + anchors_bbox_cy

        predict_x1 = predict_x - 0.5 * predict_w
        predict_y1 = predict_y - 0.5 * predict_h
        predict_x2 = predict_x + 0.5 * predict_w
        predict_y2 = predict_y + 0.5 * predict_h

        predict_boxes = torch.stack((predict_x1, predict_y1, predict_x2, predict_y2)).t()
        predict_iou = onehot_iou(anchors_bbox, predict_boxes)

        labels = torch.cat(labels, dim=0) * predict_iou
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())
        
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
        return objectness_loss, box_loss
