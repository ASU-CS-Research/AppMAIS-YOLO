import os
from typing import Optional, Union, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models import YOLOV8Backbone, YOLOV8Detector
from keras_cv.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder
from keras_cv.layers import MultiClassNonMaxSuppression
from keras_cv import bounding_box
from keras.layers import Layer
from tqdm import tqdm
import cv2
from loguru import logger
from datetime import datetime

# NOTE: <environment>/lib/python3.9/site-packages/keras_cv/models/object_detection/yolo_v8/yolo_v8_detector.py
# currently contains a bug-- we get a TypeError when training:
# TypeError: Expected int64 passed to parameter 'y' of op 'Greater', got -1.0 of type 'float' instead.
# Error: Expected int64, but got -1.0 of type 'float'.
# To run this code, you need to go to that file and change the line:
# line 540: mask_gt = tf.reduce_all(y["boxes"] > -1.0, axis=-1, keepdims=True)
# to:
# line 540: mask_gt = tf.reduce_all(y["boxes"] > -1, axis=-1, keepdims=True)


def load_dataset(path: str) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Loads the dataset as a set of tensor images and a dictionary of the class labels and bounding boxes from the given
    path.

    Args:
        path (str): Path to the dataset directory. Expects path to contain a folder called 'images' and a
        folder called 'labels', where labels contains .txt files with the same name as the corresponding image file
        in 'images' and have the following format: <class_id> <x left> <y top> <width> <height>
    Returns:
        Tuple[tf.Tensor, Dict[str, tf.Tensor]]: a tuple of images, labels where labels is a dictionary containing a
        list of bounding boxes with key 'boxes' and a list of class ids with key 'classes'
    """
    images = []
    labels = {}
    # Loop through each image file and find its corresponding label file
    for image_file in tqdm(os.listdir(os.path.join(path, 'images'))):
        image = cv2.imread(os.path.join(path, 'images', image_file))
        # You should rescale the image by 1/255.0 to normalize it here if you do not have the rescaling layer in your
        # model `image = image / 255.0`
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        images.append(image)
        with open(os.path.join(path, 'labels', image_file[:-4] + '.txt')) as f:
            boxes = []
            classes = []
            for line in f.readlines():
                class_id, x, y, w, h = line.split()
                boxes.append([float(int(x)), float(int(y)), float(int(w)), float(int(h))])
                classes.append(int(class_id))
            if 'boxes' in labels:
                labels['boxes'].append(boxes)
            else:
                labels['boxes'] = [boxes]
            if 'classes' in labels:
                labels['classes'].append(classes)
            else:
                labels['classes'] = [classes]
    # Find the max number of classifications out of any image in the dataset
    max_class_list = max([len(class_list) for class_list in labels['classes']])
    # Pad the class ids with -1 to make them all the same length (can't have jagged tensors)
    for i, class_list in enumerate(labels['classes']):
        labels['classes'][i] = np.pad(
            class_list, mode='constant', pad_width=(0, max_class_list - len(class_list)), constant_values=-1
        )
    # Pad the bounding boxes with zeros to make them all the same length
    for i, box_list in enumerate(labels['boxes']):
        labels['boxes'][i] = np.pad(
            box_list, mode='constant', pad_width=((0, max_class_list - len(box_list)), (0, 0)), constant_values=0
        )
    labels['boxes'] = tf.convert_to_tensor(labels['boxes'])
    labels['classes'] = tf.convert_to_tensor(labels['classes'])
    return tf.convert_to_tensor(images, dtype=tf.float32), labels


def load_all_datasets(path: str) -> Tuple[Tuple[tf.Tensor, Dict[str, tf.Tensor]], 
                                          Tuple[tf.Tensor, Dict[str, tf.Tensor]],
                                          Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
    """
    Loads the train, validation, and test datasets from the given path.
    recommended use: 
      (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_all_datasets(path)
    Args:
        path (str): Path to the dataset directory. Expects path to contain a folder called 'train', 'val', and 'test' 
    Returns:
        Tuple[Tuple[tf.Tensor, Dict[str, tf.Tensor]], Tuple[tf.Tensor, Dict[str, tf.Tensor]], 
              Tuple[tf.Tensor, Dict[str, tf.Tensor]]]: a tuple of train, val, and test datasets where each dataset is a
                tuple of images, labels where labels is a dictionary containing a list of bounding boxes with key 
                'boxes' and a list of class ids with key 'classes'.
    """
    train_location = os.path.join(path, 'train')
    val_location = os.path.join(path, 'val')
    test_location = os.path.join(path, 'test')
    if not os.path.exists(train_location) or not os.path.exists(val_location) or not os.path.exists(test_location):
        raise ValueError(f"Path {path} does not exist is missing one of the folders 'train', 'val', or 'test'")
    logger.debug(f'Loading train dataset from {train_location}...')
    train_ds = load_dataset(os.path.join(path, 'train'))
    logger.debug(f'Loading validation dataset from {val_location}...')
    val_ds = load_dataset(os.path.join(path, 'val'))
    logger.debug(f'Loading test dataset from {test_location}...')
    test_ds = load_dataset(os.path.join(path, 'test'))
    return train_ds, val_ds, test_ds


def build_model(num_classes: int, optimizer: tf.keras.optimizers.Optimizer, include_rescaling: Optional[bool] = False,
                classification_loss: Optional[str] = 'binary_crossentropy', box_loss: Optional[str] = 'iou',
                backbone_preset: Optional[str] = "yolo_v8_xs_backbone_coco", fpn_depth: Optional[int] = 2,
                max_anchor_matches: Optional[int] = 10) -> tf.keras.Model:
    """
    Builds a YOLOV8 model with the given input shape and number of classes.

    Args:
        num_classes (int):
        include_rescaling (Optional[bool]):
        classification_loss (Optional[str]):
        box_loss (Optional[str]):
        backbone_preset (Optional[str]):
        fpn_depth (Optional[int]):
    Returns:
        tf.keras.Model:
    """
    # pretrained backbone
    backbone = YOLOV8Backbone.from_preset(backbone_preset, include_rescaling=include_rescaling, input_shape=INPUT_SHAPE)
    # Build label encoder (responsible for transforming input boxes into trainable labels for YOLOV8Detector)
    label_encoder = YOLOV8LabelEncoder(num_classes=num_classes, max_anchor_matches=max_anchor_matches)
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
        fpn_depth=fpn_depth,
        label_encoder=label_encoder,
        prediction_decoder=prediction_decoder,
    )
    # Compile the model
    model.compile(
        classification_loss=classification_loss,
        box_loss=box_loss,
        optimizer=optimizer,
        jit_compile=False
    )
    return model


if __name__ == '__main__':
    INPUT_SHAPE = (480, 640, 3)
    BOUNDING_BOX_FORMAT = 'xywh'

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #        tf.config.set_visible_devices(gpus[1], 'GPU')
    #     except RuntimeError as e:
    #         logger.error(e)
    #         exit(1)

    num_classes = 2
    batch_size = 4
    epochs = 100
    checkpoint_path = os.path.join(os.path.abspath('./model_checkpoints/'), datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logger.info(f"Saving checkpoints to {checkpoint_path}.")
    data_path = os.path.abspath('./data/')
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_all_datasets(data_path)
    
    # logger.debug(train_labels['boxes'].shape)
    # logger.debug(train_labels['classes'].shape)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        save_weights_only=True,
        save_freq='epoch'
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0,
        restore_best_weights=True
    )

    model = build_model(num_classes=num_classes, optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                        include_rescaling=True, classification_loss='binary_crossentropy', box_loss='iou',
                        backbone_preset="yolo_v8_l_backbone_coco", fpn_depth=2, max_anchor_matches=10)

    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback],
              validation_data=(val_images, val_labels))
