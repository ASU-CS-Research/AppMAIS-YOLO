from typing import Optional, Union

import tensorflow as tf
from tensorflow import keras
from keras_cv.models import YOLOV8Backbone, YOLOV8Detector
from keras_cv.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder
from keras_cv.layers import MultiClassNonMaxSuppression
from keras_cv import bounding_box
from keras.layers import Layer
import cv2


INPUT_SHAPE = (480, 640, 3)
BOUNDING_BOX_FORMAT = "xywh"

classification_loss = "binary_crossentropy"
box_loss = 'ciou'

input_data = tf.ones(shape=((1,) + INPUT_SHAPE))
input_image = tf.ones(shape=INPUT_SHAPE).numpy()
# cv2.imwrite('./images/test_image.png', input_image)
num_classes = 2

# pretrained backbone
backbone = YOLOV8Backbone.from_preset("yolo_v8_xs_backbone_coco")
# Build label encoder (responsible for transforming input boxes into trainable labels for YOLOV8Detector)
label_encoder = YOLOV8LabelEncoder(num_classes=num_classes, max_anchor_matches=10)
label_encoder.build(input_shape=((None,) + INPUT_SHAPE))
# Prediction decoder (responsible for transforming YOLOV8 predictions into usable bounding boxes)
prediction_decoder = MultiClassNonMaxSuppression(
    bounding_box_format=BOUNDING_BOX_FORMAT,
    from_logits=True
)
model = YOLOV8Detector(
    backbone=backbone,
    num_classes=num_classes,
    bounding_box_format=BOUNDING_BOX_FORMAT,
    fpn_depth=2,
    label_encoder=label_encoder,
    prediction_decoder=prediction_decoder,
)
# outputs = model.predict(input_data)
# # outputs = model(input_data)
# # print(outputs)
# for key in outputs.keys():
#     print(key, outputs[key].shape)
# print(outputs['boxes'][0])
