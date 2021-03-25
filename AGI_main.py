
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
from utils import Normalize, pre_processing

import json # for loading label names

# torch.manual_seed(0)
# np.random.seed(0)

#%%
# 参数设置
parser = argparse.ArgumentParser(description='AGI')
parser.add_argument('--cuda', action='store_true', default=True, help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
parser.add_argument('--img', type=str, default='n01882714_11334_koala_bear.jpg', help='the images name')
parser.add_argument('--eps', type=float, default=0.05, help='epsilon value, aka step size')
parser.add_argument('--iter', type=int, default=50, help="Set the maximum number of adversarial searching iterations")
parser.add_argument('--topk', type=int, default=20, help="Set the k adversarial classes to look for")

args = parser.parse_args("") # this is only for test purpose
# args = parser.parse_args() # use this if runing as script!
# %%
epsilon = args.eps
use_cuda=args.cuda
device = torch.device("cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu")
max_iter = args.iter
topk = args.topk
selected_ids = range(0,999,int(1000/topk)) # define the ids of the selected adversarial class

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
img = cv2.imread('examples/' + args.img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# if args.model_type == 'inception':
#     # the input image's size is different
#     img = cv2.resize(img, (299, 299))

img = img.astype(np.float32) 
# img = img[:, :, (2, 1, 0)]

# %%
# set normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = Normalize(mean, std)
model = nn.Sequential(norm_layer, model).to(device)


#%%
def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    grad_lab_norm = torch.norm(data_grad_lab,p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = - data_grad_lab * delta
    return perturbed_rect, delta
    # return perturbed_image, delta

def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    c_delta = 0 # cumulative delta
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # if attack is successful, then break
        if pred.item() == targeted.item():
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        loss = output[0,targeted.item()]

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        loss_lab = output[0, init_pred.item()]
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
        c_delta += delta
    
    return c_delta, perturbed_image


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
            continue # we don't want to attack to the predicted class.

        delta, perturbed_image = pgd_step(data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy() # / topk
    img = data.squeeze().detach().cpu().numpy()
    # perturbed_image = perturbed_image.squeeze().detach().cpu().numpy()
    example = (init_pred.item(), img, adv_ex)

    # Return prediction, original image, and heatmap
    return example


# %%
# Run test
example = test(model, device, img, epsilon, topk)


# %%
# set lowerbound and upperbound for figure
percentile = 80
upperbound = 99
# input
def plot_img(plt, example):
    pred, img, ex = example
    plt.title("Pred:{}".format(class_names[pred]))
    ex = np.transpose(img, (1,2,0))
    plt.imshow(ex)

# heatmap
def plot_hm(plt, example):
    pred, img, ex = example
    # plt.title("Pred: {}".format(pred))
    plt.title("Heatmap")
    ex = np.mean(ex, axis=0)
    q = np.percentile(ex, percentile)
    u = np.percentile(ex, upperbound)
    # q=0
    ex[ex<q] = q
    ex[ex>u] = u
    ex = (ex-q)/(u-q)
    plt.imshow(ex, cmap='gray')

# input * heatmap
def plot_hm_img(plt, example):
    pred, img, ex = example
    plt.title("Input * heatmap")
    ex = np.expand_dims(np.mean(ex, axis=0), axis=0)
    q = np.percentile(ex, percentile)
    u = np.percentile(ex, upperbound)
    # q=0
    ex[ex<q] = q
    ex[ex>u] = u
    ex = (ex-q)/(u-q)
    ex = np.transpose(ex, (1,2,0))
    img = np.transpose(img, (1,2,0))

    img = img * ex
    plt.imshow(img)


# Plot several example images
plt.figure(figsize=(10,14))
for i in range(1):
    for j in range(3):
        # plot original image
        plt.subplot(1,3,1)
        plot_img(plt, example)
        plt.xticks([], [])
        plt.yticks([], [])

        # plot heatmap
        plt.subplot(1,3,2)
        plot_hm(plt, example)
        plt.xticks([], [])
        plt.yticks([], [])

        # plot heatmap * input
        plt.subplot(1,3,3)
        plot_hm_img(plt, example)
        plt.xticks([], [])
        plt.yticks([], [])

plt.tight_layout()
plt.show()

# %%
