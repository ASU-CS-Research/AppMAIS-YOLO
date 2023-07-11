from torch import nn
from ultralytics.yolo.v8.detect import DetectionPredictor
from ultralytics.yolo.utils import ops

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
                x = layer([outputs[i] for i in input_indices])
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

