# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torchstat import stat

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data import make_data_loader_stage
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model, build_detection_model_deng
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
import gc

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

from numpy import random
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

from tensorboardX import SummaryWriter


def stage_train(cfg, local_rank, distributed, logger):
    # arguments = [{}, {}, {}, {}]
    # arguments[0]["iteration"] = 0
    # arguments[1]["iteration"] = 0
    # arguments[2]["iteration"] = 0
    # arguments[3]["iteration"] = 0
    # train_data_loader = make_data_loader_stage(
    #     cfg,
    #     mode='train',
    #     is_distributed=distributed,
    #     start_iter=[item["iteration"] for item in arguments],
    # )
    writer = SummaryWriter(cfg.OUTPUT_DIR + '/runs/loss')

    debug_print(logger, 'prepare training')
    # 用于进行后面阶段学习的模型
    print("*************************")
    print("*************************")
    print("建立四个阶段的模型")
    print("*************************")
    print("*************************")
    stage_modules = []
    for stage in range(4):
        stage_model = build_detection_model_deng(cfg, False)
        stage_modules.append(stage_model)
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    print("*************************")
    print("*************************")
    print("每个阶段的模型pretrain的部分都设定城no_grad")
    print("*************************")
    print("*************************")
    eval_modules = []
    for stage in range(4):
        eval_module = (stage_modules[stage].rpn, stage_modules[stage].backbone, stage_modules[stage].roi_heads.box,)
        eval_modules.append(eval_module)
        fix_eval_modules(eval_modules[stage])

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor", ]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping[
            "roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    print("*************************")
    print("*************************")
    print("四个阶段移动到GPU")
    print("*************************")
    print("*************************")
    device = torch.device(cfg.MODEL.DEVICE)
    for stage in range(4):
        stage_modules[stage].to(device)

    print("*************************")
    print("*************************")
    print("设置optimizers和schedulers以及设定混合精度训练")
    print("*************************")
    print("*************************")
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizers = []
    schedulers = []
    for stage in range(4):
        optimizer = make_optimizer(cfg, stage_modules[stage], logger, slow_heads=slow_heads, slow_ratio=10.0,
                                   rl_factor=float(num_batch))
        optimizers.append(optimizer)
        schedulers.append(make_lr_scheduler(cfg, optimizers[stage], logger))
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    for stage in range(4):
        stage_modules[stage], optimizers[stage] = amp.initialize(stage_modules[stage], optimizers[stage],
                                                                 opt_level=amp_opt_level)
    print("*************************")
    print("*************************")
    print("设定分布并行")
    print("*************************")
    print("*************************")
    if distributed:
        for stage in range(4):
            stage_modules[stage] = torch.nn.parallel.DistributedDataParallel(
                stage_modules[stage], device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
    debug_print(logger, 'end distributed')
    arguments = [{}, {}, {}, {}]
    arguments[0]["iteration"] = 0
    arguments[1]["iteration"] = 0
    arguments[2]["iteration"] = 0
    arguments[3]["iteration"] = 0

    print("*************************")
    print("*************************")
    print("设定output dir")
    print("*************************")
    print("*************************")
    output_dirs = []
    for stage in range(4):
        output_dir = cfg.OUTPUT_DIR + "/stage" + str(stage)
        output_dirs.append(output_dir)

    print("*************************")
    print("*************************")
    print("设定checkpointer")
    print("*************************")
    print("*************************")
    save_to_disk = get_rank() == 0
    checkpointers = []
    for stage in range(4):
        checkpointer = DetectronCheckpointer(
            cfg, stage_modules[stage], optimizers[stage],
            schedulers[stage], output_dirs[stage], save_to_disk, custom_scheduler=True
        )
        checkpointers.append(checkpointer)
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    for stage in range(4):
        if checkpointers[stage].has_checkpoint():
            extra_checkpoint_data = checkpointers[stage].load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                              update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
            arguments[stage].update(extra_checkpoint_data)
        else:
            # load_mapping is only used when we init current model from detection model.
            checkpointers[stage].load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')


    # 这里获取train和validation，进去看下到底怎么分割的
    print("*************************")
    print("*************************")
    print("装training Data...")
    print("*************************")
    print("*************************")
    train_data_loader = make_data_loader_stage(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=[item["iteration"] for item in arguments],
    )

    print(train_data_loader[0].__len__())
    print(train_data_loader[1].__len__())
    print(train_data_loader[2].__len__())
    print(train_data_loader[3].__len__())

    print("*************************")
    print("*************************")
    print("装validation Data...")
    print("*************************")
    print("*************************")
    val_data_loaders = make_data_loader_stage(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    logger.info("Start training")
    print("*" * 50)
    print("每一个模型的stage0是可以复用的！！！！")
    print("如果用了confidence并且复用了stage0的checkpoint那么记得在log里面加上stage0的val结果")
    print("*" * 50)
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader[0])
    start_iter = [item["iteration"] for item in arguments]
    start_training_time = time.time()
    end = time.time()

    print_first_grad = True

    """
    测试变成parent的方法
    print(stage_modules[0].if_parent_model)
    print(stage_modules[0].roi_heads.if_parent_model)
    print(stage_modules[0].roi_heads.relation.if_parent_model)
    print(stage_modules[0].roi_heads.relation.loss_evaluator.if_parent_model)
    stage_modules[0].become_parent()
    print(stage_modules[0].if_parent_model)
    print(stage_modules[0].roi_heads.if_parent_model)
    print(stage_modules[0].roi_heads.relation.if_parent_model)
    print(stage_modules[0].roi_heads.relation.loss_evaluator.if_parent_model)
    """

    # 用于测试第一阶段训练好后接下来的训练
    # 测试验证标记符relation_train_net.py中print("得到了parent的预测进入son")
    # relation_head.py中print("parent返回预测")和print("son返回loss")
    count_for_print = 0  # 用来打进度条的不用在意！！！！

    if cfg.OUTPUT_DIR + "/param_distill":
        mkdir(cfg.OUTPUT_DIR + "/param_distill")

    parent_fisher = None  # 在stage1中计算好的fisher，用于计算stage2中参数迁移的loss
    parent_theta = None  # 用于存放parent的param

    # 设置蒸馏的温度
    for stage in range(4):
        if distributed:
            stage_modules[stage].module.set_temperature(cfg.MODEL.DISTILL_TEMPERATURE)
        else:
            stage_modules[stage].set_temperature(cfg.MODEL.DISTILL_TEMPERATURE)



    # 开始增量学习
    for stage in range(4):
        max_iter = len(train_data_loader[stage])
        if distributed:
            stage_modules[stage].module.set_stage(stage)
        else:
            stage_modules[stage].set_stage(stage)
        if stage == 0:
            stage_modules[1].cpu()
            stage_modules[2].cpu()
            stage_modules[3].cpu()
            for _, param in stage_modules[1].named_parameters():
                param.requires_grad = False
            for _, param in stage_modules[2].named_parameters():
                param.requires_grad = False
            for _, param in stage_modules[3].named_parameters():
                param.requires_grad = False
            torch.cuda.empty_cache()
        elif stage == 1:
            stage_modules[0].cuda()
            stage_modules[1].cuda()
            stage_modules[2].cpu()
            stage_modules[3].cpu()
            for _, param in stage_modules[0].named_parameters():
                param.requires_grad = False
            for _, param in stage_modules[1].named_parameters():
                param.requires_grad = True
            torch.cuda.empty_cache()
        elif stage == 2:
            stage_modules[0].cpu()
            stage_modules[3].cpu()
            stage_modules[1].cuda()
            stage_modules[2].cuda()
            for _, param in stage_modules[1].named_parameters():
                param.requires_grad = False
            for _, param in stage_modules[2].named_parameters():
                param.requires_grad = True
            torch.cuda.empty_cache()
        else:
            stage_modules[0].cpu()
            stage_modules[1].cpu()
            stage_modules[2].cuda()
            stage_modules[3].cuda()
            torch.cuda.empty_cache()


        # 这个是用于实现fisher模型传递的参数
        # 假设现在处于stage2

        son_fisher_last_iter = None  # 用于在stage2中上一个iter的fisher
        son_fisher_current_iter = None  # 用于计算在stage2中本次iter的fisher
        loss_last_iter = None


        for iteration, (images, targets, _) in enumerate(train_data_loader[stage], start_iter[stage]):


            # 读取之前保存好的一些模型参数
            # 在训练过程中就没必要在读取一次了
            if son_fisher_last_iter == None:
                # 训练的开始也没必要
                # 并且没必要存储theta因为模型本身自带了这个东西可以后面直接读出来
                if stage > 0:
                    parent_fisher = torch.load(
                        cfg.OUTPUT_DIR + "/param_distill/parent_fisher_stage" + str(stage - 1) + ".pytorch").cuda()
                    parent_theta = torch.load(
                        cfg.OUTPUT_DIR + "/param_distill/parent_theta_stage" + str(stage - 1) + ".pytorch").cuda()
                    if iteration != 0:
                        son_fisher_last_iter = torch.load(
                            cfg.OUTPUT_DIR + "/param_distill/son_fisher_current_iter(stage" + str(stage) + "iter" + str(
                                iteration) + ").pytorch").cuda()
                        loss_last_iter = torch.load(
                            cfg.OUTPUT_DIR + "/param_distill/loss_last_iter(stage" + str(stage) + "iter" + str(
                                iteration) + ").pytorch").cuda()

                if stage == 0:
                    if iteration != 0:
                        son_fisher_last_iter = torch.load(
                            cfg.OUTPUT_DIR + "/param_distill/son_fisher_current_iter(stage" + str(stage) + "iter" + str(
                                iteration) + ").pytorch").cuda()
                        loss_last_iter = torch.load(
                            cfg.OUTPUT_DIR + "/param_distill/loss_last_iter(stage" + str(stage) + "iter" + str(
                                iteration) + ").pytorch").cuda()



            if any(len(target) < 1 for target in targets):
                logger.error(
                    f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
            data_time = time.time() - end
            iteration = iteration + 1
            arguments[stage]["iteration"] = iteration

            stage_modules[stage].train()
            fix_eval_modules(eval_modules[stage])

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            """
            获取target中包含的label和predicate的方法
            print(targets[0].bbox.shape[0])
            print(targets[0].get_field("relation"))
            print(targets[0].get_field("attributes"))
            print(targets[0].get_field("labels"))
            """
            if stage == 0:
                loss_dict = stage_modules[stage](images, targets)
                # 用于记录当前loss和上一个iteration的loss的差异
                loss_current_iter = loss_dict['loss_rel'].clone().detach()
            else:
                passdown_refine_obj_logits, passdown_relation_logits = \
                    stage_modules[stage - 1](images, targets)

                """
                print("parent模型对于此次图片的预测+图片一同放入本次训练的模型中")
                print(len(passdown_relation_logits))
                print(passdown_relation_logits[0].shape)
                print(passdown_relation_logits[0])
                print(len(passdown_refine_obj_logits))
                print(passdown_refine_obj_logits[0].shape)
                print(passdown_refine_obj_logits[0])
                print("得到了parent的预测进入son")
                """

                loss_dict = stage_modules[stage](images, targets, passdown_refine_obj_logits,
                                                 passdown_relation_logits)

                # loss_dict = stage_modules[stage](images, targets)

                # 用于记录当前loss和上一个iteration的loss的差异
                loss_current_iter = loss_dict['loss_rel'].clone().detach()
                # 本次训练的参数储存
                theta_now = torch.tensor([]).cuda()

                for name, param in stage_modules[stage].named_parameters():  # 把本阶段的参数储存起来
                    if 'roi_heads.relation.predictor.context_layer' in name or \
                            'roi_heads.relation.predictor.post' in name or \
                            'roi_heads.relation.predictor.rel' in name or \
                            'roi_heads.relation.predictor.ctx' in name:
                        theta_now = torch.cat((theta_now, param.view(-1)))

                # 选贼合适的蒸馏方式
                if cfg.SOLVER.DISTILL_TYPE == 'l1':
                    loss_dict["loss_param_distill"] = cfg.SOLVER.LAMBDA_FOR_PARAM * torch.mean(parent_fisher * torch.abs(parent_theta - theta_now))
                else:
                    loss_dict["loss_param_distill"] = cfg.SOLVER.LAMBDA_FOR_PARAM * torch.mean(parent_fisher * (parent_theta - theta_now) ** 2)

                del theta_now
                torch.cuda.empty_cache()



            losses = sum(loss for loss in loss_dict.values())
            writer.add_scalar(str(stage) + ": sum of loss", losses, global_step=iteration)
            writer.add_scalar(str(stage) + ": loss_rel", loss_dict["loss_rel"], global_step=iteration)
            writer.add_scalar(str(stage) + ": loss_previous_rel", loss_dict["loss_previous_rel"], global_step=iteration)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)


            optimizers[stage].zero_grad()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizers[stage]) as scaled_losses:
                scaled_losses.backward()



# --------------------------------backward之后有梯度就可以算fisher了------------------------------------#
            # backward之后才有梯度，所以fisher的计算和更新都是在后面进行的
            son_fisher_current_iter = torch.tensor([]).cuda()
            for name, param in stage_modules[stage].named_parameters():
                if 'roi_heads.relation.predictor.context_layer' in name or \
                        'roi_heads.relation.predictor.post' in name or \
                        'roi_heads.relation.predictor.rel' in name or \
                        'roi_heads.relation.predictor.ctx' in name:
                    if param.grad != None:
                        son_fisher_current_iter = torch.cat((son_fisher_current_iter, param.grad.view(-1) ** 2))

                    else:
                        son_fisher_current_iter = torch.cat(
                            (son_fisher_current_iter, torch.tensor([0 for i in range(len(param.view(-1)))]).cuda()))
            # 清理fisher，替换nan，fisher中太大的删除，太小的赋值
            son_fisher_current_iter[son_fisher_current_iter.ne(son_fisher_current_iter)] = 0
            son_fisher_current_iter[son_fisher_current_iter > 1] = 1
            son_fisher_current_iter[son_fisher_current_iter < 1e-20] = 1e-20


            # 更新fisher matrix
            if son_fisher_last_iter == None:
                son_fisher_last_iter = torch.tensor([0 for i in range(len(son_fisher_current_iter))]).cuda()
            else:
                if (loss_last_iter - loss_current_iter) > 0:
                    son_fisher_current_iter = cfg.SOLVER.ALPHA_FOR_FISHER * son_fisher_current_iter + (
                                1 - cfg.SOLVER.ALPHA_FOR_FISHER) * son_fisher_last_iter
                    son_fisher_last_iter = son_fisher_current_iter


# --------------------------------backward之后有梯度就可以算fisher了------------------------------------#

            # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
            verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad  # print grad or not
            print_first_grad = False
            clip_grad_norm([(n, p) for n, p in stage_modules[stage].named_parameters() if p.requires_grad],
                           max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

            optimizers[stage].step()

# -----------------------------------step之后模型参数才进行更新----------------------------------------#
            # 记录本次iter的loss
            loss_last_iter = loss_dict['loss_rel'].clone().detach()
            # print("保存的" + str(loss_last_iter))
# -----------------------------------step之后模型参数才进行更新----------------------------------------#

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            iter_list = [8000, 8000, 14000]
            eta_seconds = meters.time.global_avg * (30000 - iteration - sum(iter_list[: stage]))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 200 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizers[stage].param_groups[-1]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                count_for_print = 0
            count_for_print += 1

            if (count_for_print / 2) % 10 == 0:
                print('\r 中200次循环以及进行了... %.2f %%' % (count_for_print / 200 * 100), end='')
                print("在stage" + str(stage) + "中", end='')

            if iteration % checkpoint_period == 0 or iteration == 7990:
                checkpointers[stage].save("model_{:07d}".format(iteration), **arguments[stage])

                assert np.nan not in (son_fisher_current_iter) and np.inf not in (son_fisher_current_iter)


                torch.save(son_fisher_current_iter,
                           cfg.OUTPUT_DIR + "/param_distill/son_fisher_current_iter(stage" + str(stage) + "iter" + str(
                               iteration) + ").pytorch")
                torch.save(loss_last_iter,
                           cfg.OUTPUT_DIR + "/param_distill/loss_last_iter(stage" + str(stage) + "iter" + str(
                               iteration) + ").pytorch")

            if iteration == max_iter:
                checkpointers[stage].save("model_final", **arguments[stage])

            val_result = None  # used for scheduler updating
            if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
                logger.info("Start validating")
                logger.info("In stage" + str(stage) + " iteration" + str(iteration))
                val_result = run_val(cfg, stage_modules[stage], val_data_loaders, distributed, logger)
                logger.info("Validation Result: %.4f" % val_result)

                # 找到前后两个模型的对于每个predicate的recall
                if stage != 0 and cfg.MODEL.USE_CONFIDENCE == True:
                    previous_recall, current_recall = read_recall_as_importance(cfg.OUTPUT_DIR + "/log.txt", stage,
                                                                                iteration)
                    logger.info("comparison between previous and current recall for each predicate")
                    if iteration != 8000:
                        logger.info("used for stage " + str(stage) + " iteration " + str(iteration) + "~" + str(
                            iteration + 1000))
                    else:
                        logger.info("confience for stage " + str(stage + 1) + " iteration " + str(0) + "~" + str(
                            1000) + "will be reset")
                    logger.info(previous_recall)
                    logger.info(current_recall)
                    logger.info([pre / cur if cur != 0 else 0 for pre, cur in zip(previous_recall, current_recall)])
                    if distributed:
                        stage_modules[stage].module.confidence_for_distillation(previous_recall, current_recall)
                    else:
                        stage_modules[stage].confidence_for_distillation(previous_recall, current_recall)

            # scheduler should be called after optimizer.step() in pytorch>=1.1.0
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
                schedulers[stage].step(val_result, epoch=iteration)
                if schedulers[stage].stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                    logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                    break
            else:
                schedulers[stage].step()

# --------------------------------一个stage结束后需要进行的操作--------------------------------------#

        # 本次训练的模型变成parent,同时由于跳出循环，上一个模型已经不是training状态需要重新设置
        if stage != 3:  # 最后一个模型不用变成parent
            if distributed:
                stage_modules[stage].module.become_parent()
            else:
                stage_modules[stage].become_parent()

        # 本次训练的模型变成parent之后就不要gradient了
        for _, param in stage_modules[stage].named_parameters():
            param.requires_grad = False


        # 大前提是本次stage结束不是因为本阶段训练过直接跳过
        if son_fisher_current_iter != None:
            assert np.nan not in (son_fisher_current_iter) and \
                   np.inf not in (son_fisher_current_iter)
            # stage结束一系列的储存
            # 把本阶段的fisher给保存起来

            son_fisher_current_iter = torch.log10(son_fisher_current_iter.detach())
            son_fisher_current_iter = son_fisher_current_iter - son_fisher_current_iter.mean()
            parent_fisher = 1.0 / (1.0 + torch.exp(-son_fisher_current_iter))
            parent_fisher = parent_fisher.cpu()
            # 把本阶段的参数储存起来
            parent_theta = torch.tensor([]).cuda()
            for name, param in stage_modules[stage].named_parameters():
                if 'roi_heads.relation.predictor.context_layer' in name or \
                        'roi_heads.relation.predictor.post' in name or \
                        'roi_heads.relation.predictor.rel' in name or \
                        'roi_heads.relation.predictor.ctx' in name:
                    parent_theta = torch.cat((parent_theta, param.view(-1)))
            parent_theta = parent_theta.cpu()


            assert np.nan not in (son_fisher_current_iter) and \
                   np.inf not in (son_fisher_current_iter)
            # 保存起来后面万一要用
            torch.save(parent_theta, cfg.OUTPUT_DIR + "/param_distill/parent_theta_stage" + str(stage) + ".pytorch")
            torch.save(parent_fisher, cfg.OUTPUT_DIR + "/param_distill/parent_fisher_stage" + str(stage) + ".pytorch")
            logger.info("parent importance theta fisher are saved!")


        # 删除本阶段的所有变量
        del son_fisher_last_iter
        del son_fisher_current_iter
        del loss_last_iter
        torch.cuda.empty_cache()


    # --------------------------------一个stage结束后需要进行的操作--------------------------------------#

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (4 * max_iter)
        )
    )
    return stage_modules[3]


def read_recall_as_importance(log_file_path, current_stage, current_iter):
    # 之前每一类的recall
    previous_recall_of_each_predicate = []
    # 当前现在这个模型每一个类别的recall
    current_recall_of_each_predicate = []

    with open(log_file_path) as lines:
        find_old = -1
        find_current = -1
        for line in lines:
            # 找到开始目标，倒计时开始
            if ("In stage" + str(current_stage - 1) + " iteration" + str(8000)) in line:
                find_old = 14
            # 找到打印每一个predicate的那一行
            if find_old == 0:
                parts = line.split(")")
                for i in range(len(parts) - 1):
                    previous_recall_of_each_predicate.append(float(parts[i].split(":")[1]))

            # 找到现在模型每个predicate的准度
            if ("In stage" + str(current_stage) + " iteration" + str(current_iter)) in line:
                find_current = 14
            if find_current == 0:
                parts = line.split(")")
                for i in range(len(parts) - 1):
                    current_recall_of_each_predicate.append(float(parts[i].split(":")[1]))

            find_old -= 1
            find_current -= 1

    return previous_recall_of_each_predicate, current_recall_of_each_predicate


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False


def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result


def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--resume",
        help="if t"
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # output_dir = cfg.OUTPUT_DIR
    # if output_dir:
    #     mkdir(output_dir)

    for stage in range(4):
        output_dir = cfg.OUTPUT_DIR + "/stage" + str(stage)
        if output_dir:
            mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    # 切换前面的mkdir的函数
    # model = train(cfg, args.local_rank, args.distributed, logger)
    model = stage_train(cfg, args.local_rank, args.distributed, logger)
    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    main()
