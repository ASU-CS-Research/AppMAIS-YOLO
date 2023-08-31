import os, sys

import numpy as np

sys.path.append(os.path.abspath('/home/obrienwr/AppMAIS-YOLO/yolo_v8_cam/'))

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
import ultralytics
import cv2
import matplotlib.pyplot as plt

pretrained_weights = os.path.abspath("/home/bee/bee-detection/trained_on_11r_2022.pt")
image_path = os.path.abspath('/home/obrienwr/AppMAIS-YOLO/test_images/2.jpg')
model = ultralytics.YOLO(model=pretrained_weights)
target_layers =[model.model.model[-4]]
cam = EigenCAM(model, target_layers, task='od')
image = cv2.imread(image_path)
rgb_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
rgb_img = np.float32(rgb_img) / 255

grayscale_cam = cam(rgb_img)
eigen_cams = []
for i in range(grayscale_cam.shape[2]):
    cam_image = show_cam_on_image(rgb_img, grayscale_cam[0, :, i], use_rgb=True)
    eigen_cams.append(cam_image)

fig, axes = plt.subplots(1, len(eigen_cams) + 1)
for i, ax in enumerate(axes):
    if ax == axes[-1]:
        ax.imshow(rgb_img)
        ax.set_title('Original Image')
    else:
        ax.imshow(eigen_cams[i])
        ax.set_title(f'Eigen CAM {i}')
    ax.set_xticks([])
    ax.set_yticks([])

# grayscale_cam = cam(rgb_img)[0, :, :]
# grayscale_cam = np.transpose(grayscale_cam, (1, 0))
# cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#
# # cam = EigenCAM(model, target_layers, task='od')
# # grayscale_cam = cam(rgb_img)[0, :, :]
# # grayscale_cam = np.transpose(grayscale_cam, (1, 0))
# # cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#
# # Generate predictions for image
# predictions = model(image_path)[0]
# predicted_image = predictions.plot()
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
# ax1.set_title('predicted image')
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax2.imshow(cam_img)
# ax2.set_title('cam')
# ax2.set_xticks([])
# ax2.set_yticks([])
# Save figure as a png file
fig.savefig('eigen_cam.png', bbox_inches='tight')


