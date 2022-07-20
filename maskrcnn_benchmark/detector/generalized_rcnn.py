# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads_deng, build_roi_heads

class GeneralizedRCNN_deng(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, if_parent_model):
        super(GeneralizedRCNN_deng, self).__init__()
        print("构建Generalized_rcnn")
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads_deng(cfg, self.backbone.out_channels, if_parent_model)
        self.if_parent_model = if_parent_model

    def become_parent(self):
        self.if_parent_model = True
        self.training = True
        self.roi_heads.become_parent()
    def set_stage(self, stage):
        self.stage = stage
        self.roi_heads.set_stage(stage)
    def confidence_for_distillation(self, previous_recall, current_recall):
        self.roi_heads.confidence_for_distillation(previous_recall, current_recall)
    def set_temperature(self, temperature):
        self.roi_heads.set_temperature(temperature)

    def forward(self, images, targets=None, previous_refine_obj_logits=None,
                previous_relation_logits=None, delta_theta=None, fisher_matrix=None,
                importance_scores=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
                :param delta_theta:
                :param fisher_matrix:
                :param importance_scores:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses, passdown_refine_obj_logits, passdown_relation_logits \
                = self.roi_heads(features, proposals, targets, logger,
                                 previous_refine_obj_logits, previous_relation_logits,
                                 delta_theta, fisher_matrix, importance_scores)

            # parent阶段返回对于relation和object的预测
            if self.if_parent_model:
                return passdown_refine_obj_logits, passdown_relation_logits
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        # 训练阶段返回loss
        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss.
                losses.update(proposal_losses)
            return losses

        # 测试阶段直接返回结果是BoxList
        return result










class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        return result
