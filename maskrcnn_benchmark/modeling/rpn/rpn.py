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

# add by zzf 
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
#writer = SummaryWriter('./log')

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

        

        layer_features = cfg.MODEL.RPN.CONV_LAYERS
        

        self.Unet0 = nn.Conv2d(in_channels // 4, layer_features // 2, 1)

        self.Unet1 = UnetConv2(layer_features // 2, layer_features, 0, 3, 1, 1)
        self.Unet2 = UnetConv2(layer_features, layer_features*2, 0, 3, 1, 1)
        self.Unet3 = UnetConv2(layer_features*2, layer_features*4, 0, 3, 1, 1)
        self.Unet4 = UnetConv2(layer_features*4, layer_features*8, 0, 3, 1, 1)
        self.Unet5 = UnetConv2(layer_features*12, layer_features*2, 0, 3, 1, 1)
        self.Unet6 = UnetConv2(layer_features*4, layer_features, 0, 3, 1, 1)
        self.Unet7 = UnetConv2(layer_features*2, layer_features, 0, 3, 1, 1)

        self.Unet8 = UnetConv2(layer_features, classesNum-1, 0, 3, 1, 1)

        self.GridAttention1 = _GridAttentionBlockND(layer_features*4, layer_features*8)
        self.GridAttention2 = _GridAttentionBlockND(layer_features*2, layer_features*2)
        self.GridAttention3 = _GridAttentionBlockND(layer_features, layer_features)

        self.MaxPool1 = nn.MaxPool2d(2, 2)
        self.MaxPool2 = nn.MaxPool2d(2, 2)
        self.MaxPool3 = nn.MaxPool2d(2, 2)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        '''

        layers = cfg.MODEL.RPN.CONV_LAYERS
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
        '''
    

    def forward(self, x):
        '''
        mask_feature = []
        for layer_idx, feature in enumerate(x,1):
            for layer_name in self.blocks:
                feature = F.relu(getattr(self, layer_name)(feature))
                #img_grid = vutils.make_grid(feature, normalize=True, scale_each=True, nrow=4)
                #writer.add_image(f'feature_maps_{layer_idx}', img_grid, global_step=0)
            mask_feature.append(feature)
        
        return  mask_feature
        
        '''
        mask_feature = []
        for feature in x:
            feature0 = self.Unet0(feature)
            feature1 = self.Unet1(feature0)
            feature2 = self.Unet2(self.MaxPool1(feature1))
            feature3 = self.Unet3(self.MaxPool2(feature2))
            feature4 = self.Unet4(self.MaxPool3(feature3))

            feature5 = torch.cat((self.GridAttention1(feature3, feature4), self.upsample1(feature4)), 1)
            feature5 = self.Unet5(feature5)

            feature6 = torch.cat((self.GridAttention2(feature2, feature5), self.upsample2(feature5)), 1)
            feature6 = self.Unet6(feature6)

            feature7 = torch.cat((self.GridAttention3(feature1, feature6), self.upsample3(feature6)), 1)
            feature7 = self.Unet7(feature7)

            feature8 = self.Unet8(feature7)
            
            mask_feature.append(feature8)

        return mask_feature
        

class UnetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=3, padding_size=1, init_stride=1):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            #import pdb; pdb.set_trace()
            for m0 in m:
                if isinstance(m0, nn.Conv2d):
                    nn.init.kaiming_normal_(m0.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(m0.bias, 0)

    def __getitem__(self, idx):
        if idx==0:
            return self.conv1
        elif idx==1:
            return self.conv2
            


    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation',
                 sub_sample_factor=2):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        '''
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension
        '''
        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            #bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            #bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            #bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m0.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m0.bias, 0)
            else:
                for m0 in m:
                    if isinstance(m0, nn.Conv2d):
                        nn.init.kaiming_normal_(m0.weight, mode="fan_out", nonlinearity="relu")
                        nn.init.constant_(m0.bias, 0)

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def __getitem__(self, idx):
        if idx==0:
            return self.W
        elif idx==1:
            return self.theta
        elif idx ==2:
            return self.phi
        elif idx == 3:
            return self.psi


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


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
        
        assert len(features) == 2, "error feature maps num!"
        
        features_instance = [features[0]]
        features_head = [features[1]]
        
        objectness, rpn_box_regression = self.head(features_head)
        mask_logit = None
        if self.cfg.MODEL.RPN.USE_INSTANCE:
            mask_logit = self.instance(features_instance)

        anchors = self.anchor_generator(images, features_head)

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
