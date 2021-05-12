# %%
# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
# from models.resnet import resnet20
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from utils import Normalize, pre_processing, pgd_step
import pickle

from evaluation import CausalMetric, auc, gkern

from captum.attr import IntegratedGradients

import json

# torch.manual_seed(0)
# np.random.seed(0)

img_list = []
for file in os.listdir("examples/"):
    if file.endswith(".JPEG"):
        img_list.append(file)

#%%
# 参数设置
parser = argparse.ArgumentParser(description='integrated-gradients')
parser.add_argument('--cuda', action='store_true', default=True, help='if use the cuda to do the accelartion')
parser.add_argument('--topk', type=int, default=20, help="Set the k adversarial classes to look for")
parser.add_argument('--eps', type=float, default=0.1, help='epsilon value, aka step size')
parser.add_argument('--model-type', type=str, default='resnet152', help='the type of network')


args = parser.parse_args("") # this is only for test purpose
# args = parser.parse_args() # use this if runing as script!
# %%
# 参数读取
use_cuda=args.cuda
epsilon = args.eps
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
topk = args.topk

class_idx = json.load(open("imagenet_class_index.json"))
class_names = [class_idx[str(k)][1] for k in range(len(class_idx))]

#%%
# check if have the space to save the results
if not os.path.exists('results/'):
    os.mkdir('results/')
# if not os.path.exists('results/' + args.model_type):
#     os.mkdir('results/' + args.model_type)

#%%
# start to create models...
# 选择所用的模型
if args.model_type == 'inception':
    # model = models.inception_v3(pretrained=False, init_weights=False)
    model = models.inception_v3(pretrained=True)
elif args.model_type == 'resnet152':
    model = models.resnet152(pretrained=True)
elif args.model_type == 'resnet18':
    model = models.resnet18(pretrained=True)
elif args.model_type == 'vgg19':
    model = models.vgg19_bn(pretrained=True)
else:
    raise Exception("Model is not defined.")
model.eval()
print()

# %%
# 设置normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)


# %%
# 输入模型和数据，以及一些参数，输出（预测，原图，解释图）
def test(model, device, data, epsilon, topk):
    # Send the data and label to the device
    data = pre_processing(data, device)
    data = data.to(device)

    # integrated gradients
    ig = IntegratedGradients(model)

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # initialize the step_grad towards all target false classes
    ig_grad = 0 

    attributions = ig.attribute(data, target=init_pred, n_steps=100, internal_batch_size=20)


    adv_ex = attributions.squeeze().detach().cpu().numpy() # / topk
    img = data#.squeeze().detach().cpu().numpy()
    example = (init_pred.item(), img, adv_ex)

    # Return the accuracy and an adversarial example
    return example


# %%
examples = []
factor = 100
for idx, img_name in enumerate(img_list):
    # img = img_list[0]
    img = cv2.imread('examples/' + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if args.model_type == 'inception':
        # the input image's size is different
    img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32) 
    # img = img[:, :, (2, 1, 0)]
    example = test(model, device, img, epsilon, topk)
    examples.append(example)
    if (idx+1) % factor == 0:
        f_name = 'results/resIG1000_' + str((idx+1)//factor)+ "_.txt"
        with open(f_name, 'wb') as file:
            pickle.dump(examples, file)
        examples = []

    print("{} has been processed.".format(img_name))

# with open('results/resIG1000.txt', 'wb') as file:
#         pickle.dump(examples, file)
