import ultralytics
import os
import load_yolo_model as lym
import time

data_yaml = os.path.abspath("./data.yaml")
pretrained_weights = os.path.abspath("/home/bee/bee-detection/final_model.pt")
model = ultralytics.YOLO(model=pretrained_weights)
# model.train(data=data_yaml, epochs=200, imgsz=(640, 480), verbose=True, batch=8, lr0=0.001)
# metrics = model.val(data_yaml)
# print(metrics)
# model.export()
# Time the predictions
start = time.time()
predictions = model.predict("/home/obrienwr/AppMAIS-YOLO/videos/AppMAIS11R@2022-07-31@15-05-00presentation_with_drones.mp4",
                            conf=0.64,
                            save=True,
                            stream=True)
end = time.time()
prediction_start = time.time()
while True:
    try:
        prediction = next(predictions)
    except StopIteration:
        break
prediction_end = time.time()
print("time elapsed: ", end - start, " seconds.")
print("time elapsed for predictions: ", prediction_end - prediction_start, " seconds.")
# lym.run_predictions_on_video(
#     model=model,
#     video_filepath="/home/obrienwr/AppMAIS-YOLO/videos/AppMAIS11R@2022-07-31@15-05-00presentation_with_drones.mp4",
#     destination_video_path="./presentation_yolo_video_with_drones.mp4", show=False, batch_size=512, conf=0.64
# )
