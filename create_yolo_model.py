import ultralytics
import os
import load_yolo_model as lym

data_yaml = os.path.abspath("/home/bee/bee-detection/trained_on_11r_2022.pt")
pretrained_weights = os.path.abspath("/home/bee/bee-detection/trained_on_11r_2022.pt")
model = ultralytics.YOLO(model=pretrained_weights)
# model.train(data=data_yaml, epochs=200, imgsz=(640, 480), verbose=True, batch=8, lr0=0.001)
# metrics = model.val(data_yaml)
# print(metrics)
# model.export()

lym.run_predictions_on_video(
    model=model,
    video_filepath="/home/obrienwr/AppMAIS-YOLO/videos/AppMAIS11L@2023-06-24@18-15-00fanning.h264",
    destination_video_path="./AppMAIS11L@2023-06-24@18-15-00fanning.mp4", show=False, batch_size=512, conf=0.64
)
