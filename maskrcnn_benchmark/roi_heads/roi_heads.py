# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .attribute_head.attribute_head import build_roi_attribute_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .relation_head.relation_head import build_roi_relation_head,build_roi_relation_head_deng




class CombinedROIHeads_deng(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads, if_parent_model):
        super(CombinedROIHeads_deng, self).__init__(heads)
        self.cfg = cfg.clone()
        self.if_parent_model = if_parent_model
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def become_parent(self):
        self.if_parent_model = True
        self.training = True
        self.relation.become_parent()

    def set_stage(self, stage):
        self.stage = stage
        self.relation.set_stage(stage)

    def confidence_for_distillation(self, previous_recall, current_recall):
        self.relation.confidence_for_distillation(previous_recall, current_recall)

    def set_temperature(self, temperature):
        self.relation.set_temperature(temperature)

    def forward(self, features, proposals, targets=None, logger=None,
                previous_refine_obj_logits=None, previous_relation_logits=None,
                delta_theta=None, fisher_matrix=None, importance_scores=None):
        losses ={}
        x, detections, loss_box = self.box(features, proposals, targets)
        if not self.cfg.MODEL.RELATION_ON:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss.
            losses.update(loss_box)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        if self.cfg.MODEL.RELATION_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_relation, passdown_refine_logits, passdown_relation_logits \
                = self.relation(features, detections, targets, logger,
                                previous_refine_obj_logits, previous_relation_logits,
                                delta_theta, fisher_matrix, importance_scores)
            losses.update(loss_relation)
        # {'loss_rel': tensor(31.1630, device='cuda:0', grad_fn= < NllLossBackward >),
        # 'loss_refine_obj': tensor(0.,device='cuda:0')}
        return x, detections, losses, passdown_refine_logits, passdown_relation_logits


def build_roi_heads_deng(cfg, in_channels, if_parent_model):
    print("构建roi_heads")
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON:
        roi_heads.append(("relation", build_roi_relation_head_deng(cfg, in_channels, if_parent_model)))
    if cfg.MODEL.ATTRIBUTE_ON:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads_deng(cfg, roi_heads, if_parent_model)

    return roi_heads






def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON:
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    if cfg.MODEL.ATTRIBUTE_ON:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, logger=None):
        losses ={}
        x, detections, loss_box = self.box(features, proposals, targets)
        if not self.cfg.MODEL.RELATION_ON:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss.
            losses.update(loss_box)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        if self.cfg.MODEL.RELATION_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_relation = self.relation(features, detections, targets, logger)
            losses.update(loss_relation)

        # {'loss_rel': tensor(31.1630, device='cuda:0', grad_fn= < NllLossBackward >),
        # 'loss_refine_obj': tensor(0.,device='cuda:0')}

        return x, detections, losses