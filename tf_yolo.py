import os
from typing import Optional, Dict, Tuple, List

import numpy as np
import tensorflow as tf
from keras_cv.models import YOLOV8Backbone, YOLOV8Detector
from keras_cv.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder
from keras_cv.layers import MultiClassNonMaxSuppression
from sklearn.utils import compute_class_weight
# from tensorflow.python.keras.losses import MeanSquaredError
# from tensorflow.python.keras.losses import BinaryCrossentropy
# from keras.losses import CategoricalFocalCrossentropy
# from tensorflow.keras.losses import CategoricalCrossentropy
from tqdm import tqdm
import cv2
from loguru import logger
from datetime import datetime
import wandb
from wandb.keras import WandbCallback

class YOLOV8Utils:

    @staticmethod
    def load_dataset(path: str, shuffle: Optional[bool] = False, random_seed: Optional[int] = 42) -> \
            Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[int, float]]:
        """
        Loads the dataset as a set of tensor images and a dictionary of the class labels and bounding boxes from the given
        path.

        Args:
            path (str): Path to the dataset directory. Expects path to contain a folder called 'images' and a
              folder called 'labels', where labels contains .txt files with the same name as the corresponding image file
              in 'images' and have the following format: <class_id> <x left> <y top> <width> <height>
            shuffle (bool): Whether or not to shuffle the dataset.
            random_seed (int): Random seed to use for shuffling. Ignored if shuffle is False.
        Returns:
            Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[int, float]]: a tuple of images, labels where labels is a
              dictionary containing a list of bounding boxes with key 'boxes' and a list of class ids with key 'classes'.
              Also returns a dictionary of class weights for use in training.
        """
        images = []
        labels = {}
        max_class_id = 0
        classes_for_weight_computation = []
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
                    classes_for_weight_computation.append(int(class_id))
                    if int(class_id) > max_class_id:
                        max_class_id = int(class_id)
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
        # Pad the class ids with max_class_id + 1 to make the class id arrays all the same length
        for i, class_list in enumerate(labels['classes']):
            labels['classes'][i] = np.pad(
                class_list, mode='constant', pad_width=(0, max_class_list - len(class_list)),
                constant_values=max_class_id + 1
            )
        # Pad the bounding boxes with zeros to make them all the same length
        for i, box_list in enumerate(labels['boxes']):
            labels['boxes'][i] = np.pad(
                box_list, mode='constant', pad_width=((0, max_class_list - len(box_list)), (0, 0)), constant_values=0
            )
        labels['boxes'] = tf.convert_to_tensor(labels['boxes'])
        labels['classes'] = tf.convert_to_tensor(labels['classes'])
        # Shuffle the dataset
        if shuffle:
            images, labels['boxes'], labels['classes'] = tf.random.shuffle(images, seed=random_seed), tf.random.shuffle(
                labels['boxes'], seed=random_seed), tf.random.shuffle(labels['classes'], seed=random_seed)
        # Compute the class weights
        unique_classes = np.unique(classes_for_weight_computation)
        class_weights = compute_class_weight(
            class_weight='balanced', classes=unique_classes, y=classes_for_weight_computation
        )
        class_weights = {class_id: weight for class_id, weight in zip(unique_classes, class_weights)}
        class_weights.update({max_class_id + 1: 0.0})
        # class_weights = list(class_weights)
        # class_weights.append(0.0)
        return tf.convert_to_tensor(images, dtype=tf.float32), labels, class_weights

    @staticmethod
    def load_all_datasets(path: str, shuffle: Optional[bool] = False, random_seed: Optional[int] = 42) -> \
            Tuple[Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[int, float]],
                  Tuple[tf.Tensor, Dict[str, tf.Tensor]],
                  Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
        """
        Loads the train, validation, and test datasets from the given path.
        recommended use:
          (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_all_datasets(path)
        Args:
            path (str): Path to the dataset directory. Expects path to contain a folder called 'train', 'val', and 'test'
            shuffle (bool): Whether or not to shuffle the dataset.
            random_seed (int): Random seed to use for shuffling. Ignored if shuffle is False.
        Returns:
            Tuple[Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[int, float]], Tuple[tf.Tensor, Dict[str, tf.Tensor]],
                  Tuple[tf.Tensor, Dict[str, tf.Tensor]]]: a tuple of train, val, and test datasets where each dataset is a
                    tuple of images, labels where labels is a dictionary containing a list of bounding boxes with key
                    'boxes' and a list of class ids with key 'classes'. The training dataset also returns a dictionary of
                    class weights for use in training.
        """
        train_location = os.path.join(path, 'train')
        val_location = os.path.join(path, 'val')
        test_location = os.path.join(path, 'test')
        if not os.path.exists(train_location) or not os.path.exists(val_location) or not os.path.exists(test_location):
            raise ValueError(f"Path {path} does not exist is missing one of the folders 'train', 'val', or 'test'")
        logger.debug(f'Loading train dataset from {train_location}...')
        train_ds = YOLOV8Utils.load_dataset(os.path.join(path, 'train'), shuffle=shuffle, random_seed=random_seed)
        logger.debug(f'Loading validation dataset from {val_location}...')
        # Dropping the class weights because we don't need them for validation or testing
        val_images, val_labels, _ = YOLOV8Utils.load_dataset(os.path.join(path, 'val'), shuffle=shuffle, random_seed=random_seed)
        logger.debug(f'Loading test dataset from {test_location}...')
        test_images, test_labels, _ = YOLOV8Utils.load_dataset(os.path.join(path, 'test'),  shuffle=shuffle, random_seed=random_seed)
        return train_ds, (val_images, val_labels), (test_images, test_labels)

    @staticmethod
    def build_model(num_classes: int, optimizer: tf.keras.optimizers.Optimizer, include_rescaling: Optional[bool] = False,
                    classification_loss: Optional[str] = 'binary_crossentropy', box_loss: Optional[str] = 'iou',
                    backbone_preset: Optional[str] = None, fpn_depth: Optional[int] = 2,
                    max_anchor_matches: Optional[int] = 10, metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                    freeze_backbone: Optional[bool] = True) -> \
            tf.keras.Model:
        """
        Builds a YOLOV8 model with the given input shape and number of classes.

        Args:
            num_classes (int):
            include_rescaling (Optional[bool]):
            classification_loss (Optional[str]):
            box_loss (Optional[str]):
            backbone_preset (Optional[str]):
            fpn_depth (Optional[int]):
            max_anchor_matches (Optional[int]):
        Returns:
            tf.keras.Model:
        """
        if metrics is not None:
            logger.warning(f'Metrics passed to build_model. While I would love it if we could provide metrics to the '
                           f'YoloV8Detector model, it is not currently supported. Metrics will be ignored.')
            metrics = None
        # pretrained backbone
        backbone = YOLOV8Backbone.from_preset(
            backbone_preset, include_rescaling=include_rescaling, input_shape=INPUT_SHAPE
        )
        if freeze_backbone:
            # Freeze the backbone
            for layer in backbone.layers:
                layer.trainable = False
        # Build label encoder (responsible for transforming input boxes into trainable labels for YOLOV8Detector)
        label_encoder = YOLOV8LabelEncoder(num_classes=num_classes, max_anchor_matches=max_anchor_matches)
        label_encoder.build(input_shape=((None,) + INPUT_SHAPE))
        # Prediction decoder (responsible for transforming YOLOV8 predictions into usable bounding boxes)
        prediction_decoder = MultiClassNonMaxSuppression(
            bounding_box_format=BOUNDING_BOX_FORMAT,
            from_logits=True,
            confidence_threshold=0.9
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
            jit_compile=False,
            metrics=metrics
        )
        return model

    @staticmethod
    def run_model_on_video(video_filepath: str, model: tf.keras.Model, output_filepath: str,
                           frame_limit: Optional[int] = None):
        """
        Runs the given model on the given video file and saves the output to the given output filepath.

        Args:
            video_filepath (str): Path to the video file to run the model on.
            model (tf.keras.Model): The model to run on the video.
            output_filepath (str): Path to save the output video to.
            frame_limit (Optional[int]): The maximum number of frames to run the model on. If None, runs on all frames.
        """
        capture = cv2.VideoCapture(video_filepath)
        if not capture.isOpened():
            raise ValueError(f"Could not open video file {video_filepath}")
        frames = []
        if frame_limit is None:
            logger.info(f'Getting model predictions on each frame in {video_filepath} (this could take a while)...')
        else:
            logger.info(f'Getting model predictions on each of the first {frame_limit} frames in {video_filepath}...')

        while True:
            ret, frame = capture.read()
            if frame is None or (frame_limit is not None and len(frames) >= frame_limit):
                break
            # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        capture.release()
        # Run the model on each frame
        logger.info(f'Running model on each of the {len(frames)} frames...')
        frames_tensor = []
        for i, frame in tqdm(enumerate(frames)):
            frames_tensor.append(tf.convert_to_tensor(frame, dtype=tf.float32))
        frames_tensor = tf.stack(frames_tensor)
        boxes_and_class_ids_all_frames = model.predict(frames_tensor, verbose=0)
        boxes = boxes_and_class_ids_all_frames['boxes']
        class_ids = boxes_and_class_ids_all_frames['classes']
        confidence_scores = boxes_and_class_ids_all_frames['confidence']
        for i, (frame, boxes_for_frame, class_ids_for_frame, confidence_scores_for_frame) in \
                tqdm(enumerate(zip(frames, boxes, class_ids, confidence_scores))):
            for (x, y, w, h), class_id, confidence in \
                    zip(boxes_for_frame, class_ids_for_frame, confidence_scores_for_frame):
                print('found something...')
                x, y, w, h = int(x), int(y), int(w), int(h)
                color = (140, 230, 240) if class_id == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{class_id} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            frames[i] = frame
        # Write the frames to a video file
        logger.info(f'Writing output video to {output_filepath}...')
        video_writer = cv2.VideoWriter(
            output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0])
        )
        for frame in tqdm(frames):
            video_writer.write(frame)
        video_writer.release()


if __name__ == '__main__':
    INPUT_SHAPE = (480, 640, 3)
    BOUNDING_BOX_FORMAT = 'xywh'

    logger.info(f'Using {"GPU" if len(tf.config.list_physical_devices("GPU")) > 0 else "CPU"} to train model.')
    num_classes = 3
    batch_size = 16
    epochs = 1
    checkpoint_path = os.path.join(os.path.abspath('./model_checkpoints/'), datetime.now().strftime("%Y-%m-%d/%H-%M-%S/"))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = checkpoint_path + "checkpoint.ckpt"
    logger.info(f"Saving checkpoints to {checkpoint_path}.")
    data_path = os.path.abspath('./data/')
    yolov8_utils = YOLOV8Utils()
    (train_images, train_labels, class_weights), (val_images, val_labels), (test_images, test_labels) = \
        yolov8_utils.load_all_datasets(data_path, shuffle=True)

    backbone_preset = 'yolo_v8_xs_backbone_coco'
    # classification_loss = CategoricalFocalCrossentropy(alpha=class_weights, gamma=2.0)
    # classification_loss = CategoricalCrossentropy()
    classification_loss = 'binary_crossentropy'
    box_loss = 'iou'
    fpn_depth = 3
    max_anchor_matches = 10
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    wandb.init(
        project="yolov8",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "backbone_preset": backbone_preset,
            "classification_loss": classification_loss,
            "box_loss": box_loss,
            "fpn_depth": fpn_depth,
            "max_anchor_matches": max_anchor_matches,
            "learning_rate": learning_rate
        }
    )

    # Define Callbacks
    # This one creates a model checkpoint every epoch, only saving the best one out of the whole training run.
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch'
    )
    # This one stops training early if the validation loss stops improving.
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        restore_best_weights=True
    )
    # This one logs the training run to wandb.
    wandb_callback = WandbCallback(
        monitor='val_accuracy',
        save_graph=True,
        # training_data=(train_images, train_labels),
        # validation_data=(val_images, val_labels),
        # input_type='image',
        # output_type='label',
        anonymous='allow'
    )
    # map = tfr.keras.metrics.MeanAveragePrecisionMetric()

    # Build the model with the given hyperparameters.
    model = yolov8_utils.build_model(num_classes=num_classes, optimizer=optimizer, freeze_backbone=True,
                                     include_rescaling=True, classification_loss=classification_loss, box_loss=box_loss,
                                     backbone_preset=backbone_preset, fpn_depth=fpn_depth,
                                     max_anchor_matches=max_anchor_matches)
                        # metrics=[map])
    # Train the model with the callbacks and validation data.
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
              #class_weight=class_weights,
              callbacks=[ early_stopping_callback, wandb_callback],
              validation_data=(val_images, val_labels))

    save_path = "./model_files/model.keras"
    # model.save(save_path)
    new_model = tf.keras.models.clone_model(model)

    # Evaluate the model on a testing video.
    # video_output_location = os.path.join(os.path.abspath('./video_output/'),
    #                                      datetime.now().strftime("%Y-%m-%d"))
    # if not os.path.exists(video_output_location):
    #     os.makedirs(video_output_location)
    # testing_video = 'AppMAIS10RB@2023-06-26@11-55-00.h264'
    # video_output_filepath = os.path.join(video_output_location, os.path.basename(testing_video)[:-5] + '.mp4')
    # yolov8_utils.run_model_on_video(testing_video, model, video_output_filepath, frame_limit=None)
