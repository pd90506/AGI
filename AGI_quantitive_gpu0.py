
# %%
# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from utils import Normalize, pre_processing, pgd_step
import pickle
import random

from evaluation import CausalMetric, auc, gkern

import json # for loading label names

# torch.manual_seed(0)
# np.random.seed(0)

img_list = []
for file in os.listdir("examples/"):
    if file.endswith(".JPEG"):
        img_list.append(file)
img_list.sort() # easier to track during infering
#%%
# 参数设置
parser = argparse.ArgumentParser(description='AGI')
parser.add_argument('--cuda', action='store_true', default=True, help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='resnet152', help='the type of network')
# parser.add_argument('--img', type=str, default='n01882714_11334_koala_bear.jpg', help='the images name')
parser.add_argument('--eps', type=float, default=0.05, help='epsilon value, aka step size')
parser.add_argument('--iter', type=int, default=20, help="Set the maximum number of adversarial searching iterations")
parser.add_argument('--topk', type=int, default=5, help="Set the k adversarial classes to look for")

args = parser.parse_args("") # this is only for test purpose
# args = parser.parse_args() # use this if runing as script!
# %%
epsilon = args.eps
use_cuda=args.cuda
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
max_iter = args.iter
topk = args.topk
# selected_ids = range(0,999,int(1000/topk)) # define the ids of the selected adversarial class
selected_ids = random.sample(list(range(0,999)), topk)


class_idx = json.load(open("imagenet_class_index.json"))
class_names = [class_idx[str(k)][1] for k in range(len(class_idx))]
#%%
# check if have the space to save the results
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + args.model_type):
    os.mkdir('results/' + args.model_type)
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
# set normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)



# %%
def test(model, device, data, epsilon, topk):
    # Send the data and label to the device
    data = pre_processing(data, device)
    data = data.to(device)

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    top_ids = selected_ids # only for predefined ids
    # initialize the step_grad towards all target false classes
    step_grad = 0 
    # num_class = 1000 # number of total classes
    for l in top_ids:
        targeted = torch.tensor([l]).to(device) 
        if targeted.item() == init_pred.item():
            if l < 999:
                targeted = torch.tensor([l+1]).to(device) # replace it with l + 1
            else:
                targeted = torch.tensor([l-1]).to(device) # replace it with l + 1
            # continue # we don't want to attack to the predicted class.

        delta, perturbed_image = pgd_step(data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy() # / topk
    img = data#.squeeze().detach().cpu().numpy()
    # perturbed_image = perturbed_image.squeeze().detach().cpu().numpy()
    example = (init_pred.item(), img, adv_ex)

    # Return prediction, original image, and heatmap
    return example


# %%
# Run test
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
        f_name = 'results/resnet_5step/resnet_5subclass_' + str((idx+1)//factor)+ "_.txt"
        with open(f_name, 'wb') as file:
            pickle.dump(examples, file)
        examples = []

    print("{} has been processed.".format(img_name))

# with open('results/res1000.txt', 'wb') as file:
#     pickle.dump(examples, file)
# #%%
# with open('results/res.txt', 'wb') as file:
#     pickle.dump(examples, file)



# #%%

# with open('results/res.txt', 'rb') as file:
#     examples = pickle.load(file)

# example = examples[0]

# klen = 11
# ksig = 5
# kern = gkern(klen, ksig)

# # Function that blurs input image
# blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

# deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

# label, img, sal = example
# img = img.cpu()

# h1 = deletion.single_run(img, sal, verbose=1)
# # %%
# insertion = CausalMetric(model, 'ins', 224, substrate_fn=torch.zeros_like)
# h2 = insertion.single_run(img, sal, verbose=1)
# # %%
