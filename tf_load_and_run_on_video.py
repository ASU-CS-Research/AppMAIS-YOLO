import tensorflow as tf
from tf_yolo import YOLOV8Utils
from keras_cv.models import YOLOV8Backbone, YOLOV8Detector
import keras


if __name__ == "__main__":
    model_path = "./model_files/model.keras"
    video_path = "./videos/AppMAIS7R@2023-06-26@11-55-00.h264"
    output_path = "./output_videos/outputvideo.mp4"
    custom_objects = {
        "YOLOV8Detector":YOLOV8Detector,
        "YOLOV8Backbone":YOLOV8Backbone
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)
        yolo_helper = YOLOV8Utils()
        yolo_helper.run_model_on_video(video_path,model,output_path)