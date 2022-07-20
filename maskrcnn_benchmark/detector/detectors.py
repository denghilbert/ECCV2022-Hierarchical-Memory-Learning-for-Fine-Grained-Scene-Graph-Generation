# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN,GeneralizedRCNN_deng


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}
_DETECTION_META_ARCHITECTURES_deng = {"GeneralizedRCNN": GeneralizedRCNN_deng}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)

def build_detection_model_deng(cfg,if_parent_model):
    print("构建detectors")
    meta_arch = _DETECTION_META_ARCHITECTURES_deng[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg,if_parent_model)