import ultralytics
import os
import load_yolo_model as lym

data_yaml = os.path.abspath("data.yaml")

# pretrained_weights = os.path.abspath("runs/detect/train7/weights/best.pt")
# model11s = ultralytics.YOLO(model=pretrained_weights)
# #model.train(data='data.yaml', epochs=200, imgsz=(480, 640), verbose=True, batch=8, lr0=0.001)
# metrics11 = model11s.val(data_yaml)
# model11s.export()
#
# lym.run_predictions_on_video(model=model11s, video_filepath= os.path.abspath("videos/AppMAIS11R@2022-09-02@14-00-00.mp4"), destination_video_path= "output19.mp4", show = False)

pretrained_weights = os.path.abspath("yolov8s.pt")
model11tunes = ultralytics.YOLO(pretrained_weights)
model11tunes.tune(data=data_yaml, train_args = {"epochs": 100, "data":data_yaml, "imgsz":(480, 640)})
# metrics11tunes = model11tunes.val(data_yaml)
# model11tunes.export()

# lym.run_predictions_on_video(model=model11s, video_filepath= os.path.abspath("videos/AppMAIS11R@2022-09-02@14-00-00.mp4"), destination_video_path= "output20.mp4", show = False)

# print("\n\n\nmetrics11s: \n\n\n", metrics11)
print("\n\n\nmetrics11tunes: \n\n\n", metrics11tunes)