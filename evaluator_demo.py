
#%%
# from evaluation import CausalMetric, auc, gkern
import pickle
import torch
import torch.nn as nn
import argparse
from torchvision import models
from evaluation import CausalMetric, auc, gkern
from utils import Normalize, pre_processing, pgd_step
import pandas as pd
#%%
parser = argparse.ArgumentParser(description='AGI')
parser.add_argument('--cuda', action='store_true', default=True, help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
args = parser.parse_args("") # this is only for test purpose

use_cuda=args.cuda
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")


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

# set normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)


#%%
scores = {'del': [], 'ins': []}
# df = pd.DataFrame(columns=["del", 'ins'])

for i in range(10):
    f_name = 'results/res1000_' + str(i+1)+ "_.txt"
    with open(f_name, 'rb') as file:
        examples = pickle.load(file)

    for idx, example in enumerate(examples):
        label, img, sal = example
        img = img.cpu()

        deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)
        h1 = deletion.single_run(img, sal, verbose=1)
        scores['del'].append(h1.mean())

        insertion = CausalMetric(model, 'ins', 224, substrate_fn=torch.zeros_like)
        h2 = insertion.single_run(img, sal, verbose=1)
        scores['ins'].append(h2.mean())
        if (idx+1) % 5 == 0:
            print("iteration:{}".format(i*100+idx+1))
            curDelection = sum(scores['del'])/len(scores['del'])
            curInsertion = sum(scores['ins'])/len(scores['ins'])


            print("current deletion score: {}".format(curDelection))
            print("current insertion score: {}".format(curInsertion))
df = pd.DataFrame(scores)



# %%
