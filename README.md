# AGI method
run AGI_main.py to interpret an image in the example folder.

Some variables are defined below
``
parser.add_argument('--cuda', action='store_true', default=True, help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
parser.add_argument('--img', type=str, default='n07880968_5436_burrito.jpg', help='the images name')
parser.add_argument('--eps', type=float, default=0.05, help='epsilon value, aka step size')
parser.add_argument('--iter', type=int, default=15, help="Set the maximum number of adversarial searching iterations")
parser.add_argument('--topk', type=int, default=15, help="Set the k adversarial classes to look for")
``
