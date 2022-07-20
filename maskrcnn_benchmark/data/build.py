# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import bisect
import copy
import logging

import json
import torch
import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms
import copy

# by Jiaxin
def get_dataset_statistics(cfg):
    """
    get dataset statistics (e.g., frequency bias) from training data
    will be called to help construct FrequencyBias module
    """
    logger = logging.getLogger(__name__)
    logger.info('-'*100)
    logger.info('get dataset statistics...')
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_names = cfg.DATASETS.TRAIN

    data_statistics_name = ''.join(dataset_names) + '_statistics'
    save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
    
    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-'*100)
        return torch.load(save_file, map_location=torch.device("cpu"))

    statistics = []
    for dataset_name in dataset_names:
        data = DatasetCatalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        dataset = factory(**args)
        statistics.append(dataset.get_statistics())
    logger.info('finish')

    assert len(statistics) == 1
    result = {
        'fg_matrix': statistics[0]['fg_matrix'],
        'pred_dist': statistics[0]['pred_dist'],
        'obj_classes': statistics[0]['obj_classes'], # must be exactly same for multiple datasets
        'rel_classes': statistics[0]['rel_classes'],
        'att_classes': statistics[0]['att_classes'],
    }
    logger.info('Save data statistics to: ' + str(save_file))
    logger.info('-'*100)
    torch.save(result, save_file)
    return result


def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name, cfg)
        #这data返回过来的就是{'factory': 'VGDataset', 'args':
        # {'img_dir': 'datasets/vg/VG_100K', 'roidb_file': 'datasets/vg/VG-SGG-with-attri.h5',
        # 'dict_file': 'datasets/vg/VG-SGG-dicts-with-attri.json',
        # 'image_file': 'datasets/vg/image_data.json', 'split': 'train',
        # 'filter_non_overlap': False, 'filter_empty_rels': True,
        # 'flip_aug': False, 'custom_eval': False, 'custom_path': '.'}}

        factory = getattr(D, data["factory"])
        #factory是通过data["factory"]中VGDataset返回出来的一个还没有赋值的空dataset

        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        #args是前面读取到的data里面args部分的东西加上transforms的配置声明，
        #注意前面一部分是由声明train/val的
        # make dataset from factory
        dataset = factory(**args)
        #返回出来的dataset是<maskrcnn_benchmark.data.datasets.visual_genome.VGDataset object at 0x7f9eab81e610>
        #这个时候dataset的长度就已经是57723了
        datasets.append(dataset)


    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

# 最原始的make_data_loader
def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0):
    #给进来获取的数据集必须是三个中的一个
    assert mode in {'train', 'val', 'test'}
    num_gpus = get_world_size()
    is_train = mode == 'train'

    # 如果是训练集，那么需要分割得到相应的数据集，val=5000。shuffle看情况
    if is_train:

        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    #给你个提示p用没有
    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )

    #这里连接的是maskrcnn_benchmark/config/paths_catalog.py
    DatasetCatalog = paths_catalog.DatasetCatalog

    if mode == 'train':
        #这里的dataset_list={"VG_stanford_filtered_with_attribute_train"}
        dataset_list = cfg.DATASETS.TRAIN
    elif mode == 'val':
        dataset_list = cfg.DATASETS.VAL
    else:
        dataset_list = cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    #进去得到一个对于图片的最开始变化处理的配置
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    #输入cfg，dataset的名字，变化方式，代码中记录的dataset的log，和是不是在train阶段的True/False

    #这里可以点进去可以看一下，从VG里面读取出来trainingset
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)

    #返回出来的这个datasets是个VGDataset，出来是57723个，
    #应该是为了要根据maxiter和batch重新搞一下决定跑多少个epoch
    # print("datasets")
    # print(datasets)
    # print(datasets[0])
    # print(datasets[0].__len__())
    # print(datasets[0].__getitem__(0)[0].shape)
    # print(datasets[0].__getitem__(0)[1])
    # [ < maskrcnn_benchmark.data.datasets.visual_genome.VGDataset object at 0x7f79dfca6b50 >]
    # < maskrcnn_benchmark.data.datasets.visual_genome.VGDataset object at 0x7f79dfca6b50 >
    # 57723
    # torch.Size([3, 600, 800])
    # BoxList(num_boxes=14, image_width=800, image_height=600, mode=xyxy)

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    # if mode == 'test':
    #     datasets[0].print_all_info()
    # if mode == "train":
    #     datasets[0].print_all_info()
    #     datasets[0].rerank()

    data_loaders = []
    for dataset in datasets:

        # 在这之前先调用我在VGDatasets中写好的rerank函数进行重排序后面直接用就是了
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        # sampler = make_data_sampler(dataset, False, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

        # the dataset information used for scene graph detection on customized images
        if cfg.TEST.CUSTUM_EVAL:
            custom_data_info = {}
            custom_data_info['idx_to_files'] = dataset.custom_files
            custom_data_info['ind_to_classes'] = dataset.ind_to_classes
            custom_data_info['ind_to_predicates'] = dataset.ind_to_predicates

            if not os.path.exists(cfg.DETECTED_SGG_DIR):
                os.makedirs(cfg.DETECTED_SGG_DIR)

            with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json'), 'w') as outfile:  
                json.dump(custom_data_info, outfile)
            print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json')) + ' SAVED !')
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1

        # print("-------------------------")
        # print("-------------------------")
        # print("-------------------------")
        # print("-------------------------")
        # # 这里的dataset类型是VGDataset
        # print("返回值在if is_train:里面的")
        # print(data_loaders[0])
        # print(len(data_loaders[0]))
        # print("-------------------------")
        # print("-------------------------")
        # print("-------------------------")
        # print("-------------------------")
        return data_loaders[0]

    # print("-------------------------")
    # print("-------------------------")
    # print("-------------------------")
    # print("-------------------------")
    # # 这里的dataset类型是VGDataset
    # print("返回值在if is_train:外面的")
    # print(data_loaders[0])
    # print(len(data_loaders[0]))
    # print("-------------------------")
    # print("-------------------------")
    # print("-------------------------")
    # print("-------------------------")
    return data_loaders

# 分阶段的make_data_loader
def make_data_loader_stage(cfg, mode='train', is_distributed=False, start_iter=None):
    if start_iter is None:
        start_iter = [0, 0, 0, 0]
    #给进来获取的数据集必须是三个中的一个
    assert mode in {'train', 'val', 'test'}
    num_gpus = get_world_size()
    is_train = mode == 'train'

    # 如果是训练集，那么需要分割得到相应的数据集，val=5000。shuffle看情况
    if is_train:

        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter =  [0, 0, 0, 0]


    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )

    #这里连接的是maskrcnn_benchmark/config/paths_catalog.py
    DatasetCatalog = paths_catalog.DatasetCatalog

    if mode == 'train':
        #这里的dataset_list={"VG_stanford_filtered_with_attribute_train"}
        dataset_list = cfg.DATASETS.TRAIN
    elif mode == 'val':
        dataset_list = cfg.DATASETS.VAL
    else:
        dataset_list = cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    #进去得到一个对于图片的最开始变化处理的配置
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    #输入cfg，dataset的名字，变化方式，代码中记录的dataset的log，和是不是在train阶段的True/False

    #这里可以点进去可以看一下，从VG里面读取出来trainingset
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)
    #返回出来的这个datasets是个VGDataset，出来是57723个，



    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    # if mode == 'val':
    #     datasets[0].val_debug()

    if mode == 'train':
        stage2 = copy.deepcopy(datasets[0])
        stage3 = copy.deepcopy(datasets[0])
        stage4 = copy.deepcopy(datasets[0])
        datasets.append(stage2)
        datasets.append(stage3)
        datasets.append(stage4)
        # datasets[0].rerank_by_stage([0, 10])
        # datasets[1].rerank_by_stage([10, 20])
        # datasets[2].rerank_by_stage([20, 35])
        # datasets[3].rerank_by_stage([35, 50])



        # datasets[0].rerank_only_certain_predicate_exist([0, 10])
        # datasets[1].rerank_only_certain_predicate_exist([10, 20])
        # datasets[2].rerank_only_certain_predicate_exist([20, 35])
        # datasets[3].rerank_only_certain_predicate_exist([35, 50])

        # datasets[0].pick_in_VG([ 1, 2, 3, 4, 5])

        # 这个是两个阶段用于训练的
        datasets[0].rerank_only_certain_predicate_exist([0, 8])
        # datasets[0].abalation4rebuttal()
        # datasets[1].abalation4rebuttal()
        datasets[1].rerank_only_certain_predicate_exist([8, 50])
        datasets[2].rerank_only_certain_predicate_exist([35, 50])
        datasets[3].rerank_only_certain_predicate_exist([35, 50])


    data_loaders = []
    index = 0
    for dataset in datasets:

        # 在这之前先调用我在VGDatasets中写好的rerank函数进行重排序后面直接用就是了
        if index == 0 or index ==1:
            sampler = make_data_sampler(dataset, shuffle, is_distributed)
            # sampler = make_data_sampler(dataset, False, is_distributed)
        else:
            sampler = make_data_sampler(dataset, False, is_distributed)
        if mode == 'train':
            if index == 3 or index == 1:
                batch_sampler = make_batch_data_sampler(
                    dataset, sampler, aspect_grouping, images_per_gpu, num_iters * 2, start_iter[index]
                )
            elif index == 2:
                batch_sampler = make_batch_data_sampler(
                    dataset, sampler, aspect_grouping, images_per_gpu, num_iters + 6000, start_iter[index]
                )
            else:
                batch_sampler = make_batch_data_sampler(
                    dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter[index]
                )
        else:
            if index == 3:
                batch_sampler = make_batch_data_sampler(
                    dataset, sampler, aspect_grouping, images_per_gpu, num_iters * 2, start_iter[index]
                )
            elif index == 2:
                batch_sampler = make_batch_data_sampler(
                    dataset, sampler, aspect_grouping, images_per_gpu, num_iters + 6000, start_iter[index]
                )
            else:
                batch_sampler = make_batch_data_sampler(
                    dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter[index]
                )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

        # the dataset information used for scene graph detection on customized images
        if cfg.TEST.CUSTUM_EVAL:
            custom_data_info = {}
            custom_data_info['idx_to_files'] = dataset.custom_files
            custom_data_info['ind_to_classes'] = dataset.ind_to_classes
            custom_data_info['ind_to_predicates'] = dataset.ind_to_predicates

            if not os.path.exists(cfg.DETECTED_SGG_DIR):
                os.makedirs(cfg.DETECTED_SGG_DIR)

            with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json'), 'w') as outfile:
                json.dump(custom_data_info, outfile)
            print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json')) + ' SAVED !')
        data_loaders.append(data_loader)

        index += 1

    # if is_train:
    #     # during training, a single (possibly concatenated) data_loader is returned
    #     assert len(data_loaders) == 1
    #     return data_loaders[0]

    return data_loaders