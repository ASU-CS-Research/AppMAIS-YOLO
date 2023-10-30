import ultralytics
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
from torch import nn
import os
import torchvision.utils


def vis_tensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


model_path = os.path.abspath("/home/bee/bee-detection/trained_on_11r_2022.pt")
image_path = os.path.abspath('/home/obrienwr/AppMAIS-YOLO/test_images/AppMAIS11R@2022-10-11@14-35-00.jpg')

model = ultralytics.YOLO(model=model_path)
image = Image.open(image_path)

model_children = list(model.model.ultralytics_model[-1].children())
conv_2d_layers = []
model_weights = []
conv_2d_layers.append()
for i, child in enumerate(model_children):
    print(type(child))
    # print(list(child.children()))
    for j, grandchild in enumerate(child.children()):
        print(type(grandchild))
        print(list(grandchild.children()))
        for k, greatgrandchild in enumerate(grandchild.children()):
            if type(greatgrandchild) == nn.Conv2d:
                model_weights.append(greatgrandchild.weight)
                conv_2d_layers.append(greatgrandchild)
                print(greatgrandchild.weight.shape)
print(f'Total conv2d layers: {len(conv_2d_layers)}')
for layer in conv_2d_layers:
    print(layer)
    print(layer.weight.shape)
layer = conv_2d_layers[0]
kernels = layer.weight.detach().clone()
vis_tensor(kernels, ch=0, allkernels=False)

plt.savefig('./filter_img.png')