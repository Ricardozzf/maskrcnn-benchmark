# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .loss import make_rpn_loss_evaluator
from .loss import make_rpn_mask_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

@registry.RPN_HEADS.register("SingleConvRPNIoUHead")
class RPNIoUHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNIoUHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.iou_conv1_1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        self.iou_conv1_2 = nn.Conv2d(
            in_channels, 2* in_channels, kernel_size=1, stride=1, padding=0
        )
        
        self.iou_conv1_3 = nn.Conv2d(
            2 * in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            bbox_reg.append(self.bbox_pred(t))
            for i in range(3):
                t = F.relu(self.iou_conv1_1(t))
                t = F.relu(self.iou_conv1_2(t))
                t = F.relu(self.iou_conv1_3(t))
            logits.append(self.cls_logits(t))

        return logits, bbox_reg

@registry.INSTANCE_MASK.register("SingleConvRPNInstanceHead")
class RPNInstanceHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNInstanceHead, self).__init__()
        classesNum = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        next_feature = in_channels

        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "instanceConv_{}".format(layer_idx)
            module = nn.Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

        module = nn.Conv2d(next_feature, classesNum-1, 1, stride=1, padding=0)
        layer_name = "instance_predict"
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(module.bias, 0)
        self.add_module(layer_name, module)
        self.blocks.append(layer_name)


    def forward(self, x):
        mask_feature = []
        for feature in x:
            for layer_name in self.blocks:
                feature = F.relu(getattr(self, layer_name)(feature))
            mask_feature.append(feature)
        return  mask_feature


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        instance_mask = registry.INSTANCE_MASK[cfg.MODEL.RPN.INSTANCE_MASK]

        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )
        instance = instance_mask(cfg, in_channels)

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)
        loss_mask_evaluator = make_rpn_mask_loss_evaluator(cfg)

        self.anchor_generator = anchor_generator
        self.head = head
        self.instance = instance
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.loss_mask_evaluator = loss_mask_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        mask_logit = None
        if self.cfg.MODEL.RPN.USE_INSTANCE:
            mask_logit = self.instance(features)

        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, mask_logit,
            images, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, mask_logit, images, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        if self.cfg.MODEL.RPN.USE_INSTANCE:
            loss_rpn_mask = self.loss_mask_evaluator(images, mask_logit, targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_rpn_mask": loss_rpn_mask,
            }
        else:
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)
