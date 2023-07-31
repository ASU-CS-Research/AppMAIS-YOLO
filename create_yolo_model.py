import ultralytics
import os
import load_yolo_model as lym

data_yaml = os.path.abspath("data.yaml")
pretrained_weights = os.path.abspath("/home/obrienwr/AppMAIS-YOLO/yolov8s.pt")
model = ultralytics.YOLO(model=pretrained_weights)
model.train(data='data.yaml', epochs=200, imgsz=(640, 480), verbose=True, batch=8, lr0=0.001)
metrics = model.val(data_yaml)
print(metrics)
model.export()

# lym.run_predictions_on_video(
#     model=model,
#     video_filepath="/home/obrienwr/AppMAIS-YOLO/videos/AppMAIS1L@2023-06-26@11-55-00.h264",
#     destination_video_path="./output_AppMAIS7L@2023-06-26@11-55-00.mp4", show=False
# )
