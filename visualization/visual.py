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

print("准备！")
# detected_origin_path = '/media/data1/deng_sgg/backup/transformer-precls/inference/VG_stanford_filtered_with_attribute_test/'
detected_origin_result1 = torch.load('/media/data1/deng_sgg/stage_distill/vctree_precls_mse_nomask_3stage_fisher_lambda100_l1/param_distill/parent_theta_stage0.pytorch')
t = torch.load('/media/data1/deng_sgg/stage_distill/motifs_precls_mse_nomask_3stage/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch')

print("开始！")