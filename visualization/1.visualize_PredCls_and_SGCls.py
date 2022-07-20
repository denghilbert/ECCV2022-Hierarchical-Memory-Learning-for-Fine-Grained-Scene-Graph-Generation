#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import json
import os

from pyecharts import options as opts
from pyecharts.charts import Graph, Page
import pandas as pd

# In[4]:

print("准备！")
image_file = json.load(open('/media/data1/deng_sgg/Unbiased_SGG/datasets/vg/image_data.json'))
vocab_file = json.load(open('/media/data1/deng_sgg/Unbiased_SGG/datasets/vg/VG-SGG-dicts-with-attri.json'))
data_file = h5py.File('/media/data1/deng_sgg/Unbiased_SGG/datasets/vg/VG-SGG-with-attri.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp


# In[36]:


# load detected results
# detected_origin_path = '/media/data1/deng_sgg/stage_distill/vctree_precls_mse_nomask_3stage_fisher_lambda100_l1/inference/VG_stanford_filtered_with_attribute_test/'
detected_origin_path = '/media/data1/deng_sgg/stage_distill/motifs_precls_mse_nomask_3stage_fisher_lambda100_l1/inference/VG_stanford_filtered_with_attribute_test/'
# detected_origin_path = '/media/data1/deng_sgg/backup/VCTree-precls/inference/VG_stanford_filtered_with_attribute_test/'
# detected_origin_path = '/media/data1/deng_sgg/backup/motifs-precls/inference/VG_stanford_filtered_with_attribute_test/'
# detected_origin_path = '/media/data1/deng_sgg/backup/transformer-precls/inference/VG_stanford_filtered_with_attribute_test/'
detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))

print("开始！")
# In[42]:


# get image info by index
def get_info_by_idx(idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    # image path
    img_path = detected_info[idx]['img_file']
    # boxes
    boxes = groundtruth.bbox
    # object labels
    idx2label = vocab_file['idx_to_label']
    labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]
    pred_labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    gt_rels = groundtruth.get_field('relation_tuple').tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    #mask = pred_rel_score > thres
    #pred_rel_score = pred_rel_score[mask]
    #pred_rel_label = pred_rel_label[mask]
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    return img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label


# In[43]:


def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))
    
def draw_image(select_idx, img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    pic = Image.open(img_path)
    num_obj = boxes.shape[0]

    for i in range(num_obj):
        info = labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    if print_img:
        # pic.save("/media/data1/deng_sgg/visualized_result_without_text/" + str(select_idx) + ".jpg")
        # pic.save("/media/data1/deng_sgg/visualized_result/" + str(select_idx) + ".jpg")
        # pic.save("/media/data1/deng_sgg/visual_new/" + str(select_idx) + ".jpg")
        # pic.save("/media/data1/deng_sgg/visual_horse/" + str(select_idx) + ".jpg")
        # pic.save("/media/data1/deng_sgg/visual_playing/" + str(select_idx) + ".jpg")
        # pic.save("/media/data1/deng_sgg/visual_painted on/" + str(select_idx) + ".jpg")
        pic.save("/media/data1/deng_sgg/visual_sitting on/" + str(select_idx) + ".jpg")
    if print_img:
        print('*' * 50)
        print_list('gt_boxes', labels)
        print('*' * 50)
        print_list('gt_rels', gt_rels)
        print('*' * 50)
    print_list('pred_rels', pred_rels[:20])


    # nodes = [ {"name": node, "symbolSize": 10} for node in labels]
    #
    # links = []
    # for triplet in pred_rels[:7]:
    #     links.append({"source": triplet[0], "target": triplet[2], "label_opts": triplet[1]})
    # graph = (
    #     Graph(init_opts=opts.InitOpts(width="1000px", height="800px"))
    #         .add("",
    #              nodes,
    #              links,
    #              layout="force"
    #              )
    #         .set_global_opts(title_opts=opts.TitleOpts(title="web"))
    # )
    # graph.render("/media/data1/deng_sgg/visualized_result/" + str(select_idx) + ".html")
    print('*' * 50)
    
    return None


# In[52]:


def show_selected(idx_list):
    for select_idx in idx_list:
        print(select_idx)
        draw_image(select_idx, *get_info_by_idx(select_idx, detected_origin_result))
        
def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx+length):
        print(cand_idx)
        draw_image(*get_info_by_idx(cand_idx, detected_origin_result))


# In[55]:
# 用于寻找gt和pred之间错误的
def find_on(idx_list):
    for select_idx in idx_list:
        filter_on(select_idx, *get_info_by_idx(select_idx, detected_origin_result))

    return None
def filter_on(select_idx, img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    for gt_triplet in gt_rels:
        # 这个部分单纯用于寻找gt和pred有差异的on和has
        # if gt_triplet[1] == 'on':
        if gt_triplet[1] == 'has':
            for triplet in pred_rels[:20]:
                if triplet[0] == gt_triplet[0] and triplet[2] == gt_triplet[2] and triplet[1] != gt_triplet[1]:
                    print(str(select_idx) + "gt: " + str(gt_triplet) + " pre: " + str(triplet))

    #show_all(start_idx=0, length=5)

# 用于寻找informative和general的
def find_informative(idx_list):
    for select_idx in idx_list:
        filter_informative(select_idx, *get_info_by_idx(select_idx, detected_origin_result))

    return None
def filter_informative(select_idx, img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    for gt_triplet in gt_rels:
        if gt_triplet[1] == 'behind':
            print(gt_triplet)
        # if gt_triplet[1] == 'sitting on':
        #     print(select_idx, end=',')
        # for triplet in pred_rels[: 20]:
        #     if triplet[0] == gt_triplet[0] and gt_triplet[1] == 'parked on' and triplet[2] ==gt_triplet[2]:
        #         print("gt:" + str(gt_triplet))
        #         print("pred:" + str(triplet))
    # print("*" * 50)
        # # 人坐在凳子上
        # with open('/media/data1/deng_sgg/annotation_of_on.txt', 'a') as f:
        #     count_on = 0
        #     count_other = 0
        #     if ('chair' in gt_triplet[2] or 'seat' in gt_triplet[2] or 'bench' in gt_triplet[2]) and ('woman' in gt_triplet[0] or 'lady' in gt_triplet[0] or 'guy' in gt_triplet[0] or 'people' in gt_triplet[0] or 'men' in gt_triplet[0] or 'boy' in gt_triplet[0] or 'kid' in gt_triplet[0] or 'child' in gt_triplet[0] or 'man' in gt_triplet[0]):
        #         print(select_idx)
        #         print(gt_triplet)
        #         f.writelines(str(gt_triplet) + '\n')

        # 人骑车
        # with open('/media/data1/deng_sgg/annotation_of_man_ride.txt', 'a') as f:
        #     count_on = 0
        #     count_other = 0
        #     if ('motorcycle' in gt_triplet[2] or 'bike' in gt_triplet[2]) and ('woman' in gt_triplet[0] or 'lady' in gt_triplet[0] or 'guy' in gt_triplet[0] or 'people' in gt_triplet[0] or 'men' in gt_triplet[0] or 'boy' in gt_triplet[0] or 'kid' in gt_triplet[0] or 'child' in gt_triplet[0] or 'man' in gt_triplet[0]):
        #         print(select_idx)
        #         print(gt_triplet)
        #         f.writelines(str(gt_triplet) + '\n')




# 用于生成图片
# 所有猫猫的80,98,217,283,292,459,1099,1125,1170,1171,1462,1781,1892,2036,2457,2615,2827,2836,3017,3076,3158,3181,3368,3605,3704,3886,4157,4352,4380,4484,4537,4666,4773,4834,4865,4958,5063,5169,5350,5402,5442,5454,5772,5824,5885,5894,5980,6050,6126,6153,6846,7067,7257,7284,7407,7511,7657,7961,8012,8170,8234,8340,8546,8647,8852,8908,8939,8973,9112,9144,9418,9481,9565,9622,9719,9804,9909,9923,9988,10071,10162,10200,10287,10307,10338,10367,10413,10518,10785,10821,10880,10936,10940,10958,11812,11834,11986,12104,12108,12241,12424,12466,12471,12709,12784,12822,12827,12908,12917,13027,13112,13193,13273,13365,13616,13675,13713,13784,13849,13904,14070,14277,14391,14931,14933,14957,15046,15068,15143,15468,15597,15683,15697,15903,16144,16495,16845,16895,17165,17181,17535,17571,17612,17727,17767,17948,18387,18522,18530,18816,19102,19248,19267,19638,19748,20014,20021,20053,20141,20149,20523,20600,20712,20974,21061,21077,21096,21324,21401,21571,21607,21940,21973,22070,22101,22204,22233,22246,22404,22519,22542,22573,22664,22802,22805,22881,23018,23055,23211,23367,23502,23552,23703,23766,24296,24322,24350,24550,24787,24948,25216,25332,25371,25414,25506,25523,25529,25884,25948,25993,26034,26113,26184,26234,26268
# show_selected([597, 3118, 3684, 4936, 8951, 9442, 10375, 10994, 12660])
# 用于进行错误筛选
# find_on([num for num in range(26446)])
# 用于找informative的
find_informative([num for num in range(26446)])
# find_informative([2061])
# In[ ]:


def find_pic_for_paper(idx_list):
    for select_idx in idx_list:
        filter_pic_for_paper1(select_idx, *get_info_by_idx(select_idx, detected_origin_result))


    return None
def filter_pic_for_paper1(select_idx, img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    if len(boxes) <= 10 and len(boxes) > 5:
        for gt_triplet in gt_rels:
            if gt_triplet[1] == 'parked on':
                print(str(select_idx) + ",", end='')

def filter_pic_for_paper2(select_idx, img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    return None
# find_pic_for_paper([num for num in range(26446)])




def callculate_on(idx_list):
    TP, FN, FP, TN, total = 0, 0, 0, 0, 0
    for select_idx in idx_list:
        TP_t, FN_t, FP_t, TN_t, total_t = count_on(TP, FN, FP, TN, *get_info_by_idx(select_idx, detected_origin_result))
        TP += TP_t
        FN += FN_t
        FP += FP_t
        TN += TN_t
        total += total_t


    print("recall of on is:" + str(float(TP) / (total)))
    return None

def count_on(TP, FN, FP, TN, img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    TP, FN, FP, TN = 0, 0, 0, 0
    # 下面这个循环计算写成对于gt计算recall
    # for gt_triplet in gt_rels:
    #     if gt_triplet[1] == 'on':
    #         for triplet in pred_rels[:100]:
    #             if gt_triplet[0] == triplet[0] and gt_triplet[2] == triplet[2] and triplet[1] == 'on':
    #                 TP += 1
    #             if gt_triplet[0] == triplet[0] and gt_triplet[2] == triplet[2] and triplet[1] != 'on':
    #                 FN += 1
    #     if gt_triplet[1] != 'on':
    #         for triplet in pred_rels[:100]:
    #             if gt_triplet[0] == triplet[0] and gt_triplet[2] == triplet[2] and triplet[1] == 'on':
    #                 FP += 1
    #             if gt_triplet[0] == triplet[0] and gt_triplet[2] == triplet[2] and triplet[1] != 'on':
    #                 TN += 1
    TP_FN = 0
    for gt_triplet in gt_rels:
        # if gt_triplet[1] == 'on':
        # if gt_triplet[1] == 'has':
        if gt_triplet[1] == 'wearing':
            TP_FN += 1
            for triplet in pred_rels[:100]:
                # if gt_triplet == triplet:

                # if gt_triplet[0] == triplet[0] and gt_triplet[2] == triplet[2] and \
                #         (triplet[1] == 'on' or triplet[1] == 'sitting on' or triplet[1] == 'riding' or
                #          triplet[1] == 'standing on' or triplet[1] == 'walking on' or
                #          triplet[1] == 'over' or triplet[1] == 'laying on' or
                #          triplet[1] == 'parked on' or triplet[1] == 'covering' or
                #          triplet[1] == 'lying on' or triplet[1] == 'mounted on' or
                #          triplet[1] == 'growing on' or triplet[1] == 'of' or triplet[1] == 'above' or
                #          triplet[1] == 'painted on' or triplet[1] == 'to' or triplet[1] == 'part of'):


                # if gt_triplet[0] == triplet[0] and gt_triplet[2] == triplet[2] and \
                #         (triplet[1] == 'has' or triplet[1] == 'with' or triplet[1] == 'wearing' or
                #          triplet[1] == 'belonging to' or triplet[1] == 'carrying' or
                #          triplet[1] == 'using' or triplet[1] == 'attached to'):

                if gt_triplet[0] == triplet[0] and gt_triplet[2] == triplet[2] and \
                        (triplet[1] == 'wearing' or triplet[1] == 'wears'):
                    TP += 1
                    break

    # 这里前面总和和后面不相等原因是因为有的时候预测里面没有gt的head+tail组合
    return TP, FN, FP, TN, TP_FN

# 26446
# 重新计算如果吧on的子类变成on的recall是多少

print("重新计算recall")
callculate_on([num for num in range(26446)])

# has:原来0.8051 不清理0.5607 清理0.8964996022275259
# on:原来0.7924 不清理0.1932 清理0.6945083540929502
# wears:原来0.9697 不清理0.8300 清理0.9574647607702997
# 读取存储的数据统计
# val_metric = torch.load("/media/data1/deng_sgg/stage_distill/vctree_precls_distill+stage+sampling/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
# print(val_metric.keys())
# print(val_metric['predcls_mean_recall'])
# print(val_metric['predcls_mean_recall_list'])






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




