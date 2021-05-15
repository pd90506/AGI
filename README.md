# Adversarial Gradient Integration (AGI)
This is a pytorch implementation of our paper `Explaining Deep Neural Network Models with Adversarial Gradient Integration`.
One can run `AGI_main.py` to interpret an image in the example folder. The image's format must be JPEG.

Some parameters are defined below
```
parser.add_argument('--cuda', action='store_true', default=True, help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
parser.add_argument('--img', type=str, default='n07880968_5436_burrito.jpg', help='the images name')
parser.add_argument('--eps', type=float, default=0.05, help='epsilon value, aka step size')
parser.add_argument('--iter', type=int, default=15, help="Set the maximum number of adversarial searching iterations")
parser.add_argument('--k', type=int, default=15, help="Set k adversarial classes to look for")
```
In the paper, we choose `iter=20` and `k=20`.

# Examples
Below are some examples we tested on the ImageNet dataset, using the InceptionV3 pretrained classification model.
