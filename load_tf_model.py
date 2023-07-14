import tensorflow as tf
import os
from keras_cv.models import YOLOV8Backbone, YOLOV8Detector
from keras_cv.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder
from keras_cv.layers import MultiClassNonMaxSuppression
from tf_yolo import YOLOV8Utils

# path = os.path.abspath('/home/obrienwr/AppMAIS-YOLO/model_checkpoints/2023-07-14/09-04-01/model.keras')
custom_layers = {'YOLOV8Backbone': YOLOV8Backbone,
                 'YOLOV8LabelEncoder': YOLOV8LabelEncoder,
                 'MultiClassNonMaxSuppression': MultiClassNonMaxSuppression,
                 'YOLOV8Detector': YOLOV8Detector}
# # Load the model
# with tf.keras.utils.custom_object_scope(custom_layers):
#     model = tf.keras.models.load_model(path)
path = os.path.abspath('/home/obrienwr/AppMAIS-YOLO/model_checkpoints/2023-07-14/10-03-49')
json_path = os.path.join(path, 'model.json')
weights_path = os.path.join(path, 'model.h5')
with open(json_path, 'r') as json_file:
    json_model = json_file.read()
model = tf.keras.models.model_from_json(json_model, custom_objects=custom_layers)
# model.summary()
for layer in model.layers:
    layer.trainable=False
model.load_weights(weights_path)

video_path = os.path.abspath('/home/obrienwr/AppMAIS-YOLO/videos/AppMAIS14L@2023-06-26@11-55-00.h264')
YOLOV8Utils.run_model_on_video(video_filepath=video_path, model=model, output_video_filepath=os.path.join(
        './', os.path.basename(video_path)[:-5] + '_output.mp4'
    )
)
# def myprint(s):
#     with open('./model_summary_after_loading.txt', 'a') as f:
#         print(s, file=f)
#
# model.summary()
