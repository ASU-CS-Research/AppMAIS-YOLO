import torch
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from torch import nn
from ultralytics.yolo.v8.detect import DetectionPredictor
from ultralytics.yolo.utils import ops
import ultralytics
import os
import json
from torchsummary import summary

class PytorchYOLOV8(nn.Module):
    def __init__(self, sequential_model, yaml_config, classes, conf_thresh=0.4, iou_thresh=0.5, max_det=20, agnostic_nms=False):
        super(PytorchYOLOV8, self).__init__()
        self.sequential_model = sequential_model
        self.yaml_config = yaml_config
        self.classes = classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.detection_predictor = DetectionPredictor()
        # self.detection_predictor.args.classes = self.classes
        # print(self.detection_predictor.args.classes)
        # exit()

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.sequential_model):
            type = layer.type.split('.')[-1]
            # print(i, type)
            if type == 'Concat':
                if i > len(self.yaml_config['backbone']):
                    index = i - len(self.yaml_config['backbone'])
                    # print(self.yaml_config['head'][index])
                    input_indices = self.yaml_config['head'][index][0]

                else:
                    # print(self.yaml_config['backbone'][i])
                    input_indices = self.yaml_config['backbone'][i][0]
                # print(input_indices)
                # print(layer)
                # print(outputs[input_indices[0]])
                # print()
                # print(outputs[input_indices[1]])
                x = layer((outputs[input_indices[0]], outputs[input_indices[1]]))
                outputs.append(x)
                # exit()
            elif type == 'Detect':
                index = i - len(self.yaml_config['backbone'])
                input_indices = self.yaml_config['head'][index][0]
                x = layer([outputs[j] for j in input_indices])
                outputs.append(x)
            else:
                x = layer(x)
                outputs.append(x)
        return x

    def predict(self, x):
        output = self.forward(x)
        preds = ops.non_max_suppression(
            output, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh,
            classes=self.classes, agnostic=self.agnostic_nms,
            max_det=self.max_det
        )
        return preds

    def pprint_yaml(self):
        for key in self.yaml_config:
            if key == 'head' or key == 'backbone':
                print(key)
                for layer in self.yaml_config[key]:
                    print('\t', layer)
            else:
                print(key, self.yaml_config[key])


if __name__ == '__main__':
    model_path = os.path.abspath("/home/bee/bee-detection/trained_on_11r_2022.pt")
    image_path = os.path.abspath('/home/obrienwr/AppMAIS-YOLO/test_images/AppMAIS11R@2022-10-11@14-35-00.jpg')
    ultralytics_model = ultralytics.YOLO(model=model_path)
    # print(dir(ultralytics_model.model))
    # # print(ultralytics_model.model.info)
    # # print(ultralytics_model.model.named_parameters)
    # print(ultralytics_model.model.yaml)
    # # summary(ultralytics_model.model, (3, 640, 480))
    # exit()
    pytorch_sequential_model = ultralytics_model.model.__dict__["_modules"]["model"]
    pytorch_yaml_config = ultralytics_model.model.__dict__["yaml"]
    # classes = ultralytics_model.model.__dict__["names"]
    classes = [0, 1]
    model = PytorchYOLOV8(pytorch_sequential_model, pytorch_yaml_config, classes, conf_thresh=0.64)
    # model.pprint_yaml()
    model = model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    image_tensor = image_tensor.type(torch.float32)
    results = model.predict(image_tensor)
    activations = model.forward(image_tensor)

    # print(activations[1])
    # Convert the tuple activations to a tensor
    print(activations[0].shape)
    for activation_output in activations[1]:
        print(activation_output.shape)
    exit()
    # print('Results from the reconstructed pytorch model:')
    # print(results)
    # print(model)
    # print(dir(model))
    # for i, child in enumerate(model.named_children()):
    #     print(i)
    layer = next(model.named_children())[1][-1]
    # print(layer)
    target_layers = [layer]
    print(target_layers)
    cam = EigenCAM(model, target_layers, use_cuda=False)
    grayscale_cam = cam(image_tensor)[0, :, :]
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    Image.fromarray(cam_image)

    # image = cv2.imread(image_path)
    # ultralytics_results = ultralytics_model.predict(image)
    # print('Results from the ultralytics model:')
    # for box in ultralytics_results[0].boxes:
    #    x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
    #    print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}, class_id: {class_id}')

