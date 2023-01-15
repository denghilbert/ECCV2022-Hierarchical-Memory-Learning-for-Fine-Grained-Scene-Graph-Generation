# Hierarchical Memory Learning for Fine-Grained Scene Graph Generation

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.7.1-%237732a8)

Our paper [Hierarchical Memory Learning for Fine-Grained Scene Graph Generation](https://arxiv.org/abs/2203.06907) has been accepted by ECCV 2022.

## Installation

Follow this [installation](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/INSTALL.md).

## Dataset

Check [dataset](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md) for instructions of dataset preprocessing.

## Pretrained Models

You can download the [pretrained Faster R-CNN](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw) we used in the paper.

## Training for HML

I take the training PredCls for MOTIFS under HML as an example:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10030 --nproc_per_node=2 /home/dengyouming/project/HML/tools/relation_train_distill_fisher_only.py --config-file "/home/dengyouming/project/HML/configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.USE_CONFIDENCE False MODEL.DISTILL_TEMPERATURE 2 MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 1000 SOLVER.LAMBDA_FOR_PARAM 1.0 SOLVER.ALPHA_FOR_FISHER 0.5 SOLVER.DISTILL_TYPE l2 SOLVER.BASE_LR 0.001 GLOVE_DIR /home/dengyouming/project/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/dengyouming/project/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/dengyouming/project/eccv/motifs_hml
```

## Evaluation

Evaluate model with following command:

```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/test.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/dengyouming/project/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/dengyouming/project/eccv/motifs_hml OUTPUT_DIR /home/dengyouming/project/eccv/motifs_hml
```

## Citations

If you find this project helps your research, please kindly consider citing our project or papers in your publications.

```
@inproceedings{deng2022hml,
  title={Hierarchical Memory Learning for Fine-Grained Scene Graph Generation},
  author={Deng, Youming and Li, Yansheng and Zhang, Yongjun and Xiang, Xiang and Wang, Jian and Chen, Jingdong and Ma, Jiayi},
  booktitle= "European Conference on Computer Vision",
  year={2022}
}
```



## Acknowledgements

Part of our code is inherited from [Unbiased SGG](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). We are grateful to the authors for releasing their code.