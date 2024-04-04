import ultralytics
import os

model_path = os.path.abspath("./trained_models/final_model.pt")
model = ultralytics.YOLO(model_path)

video_path = os.path.abspath('videos/AppMAIS11L@2023-06-24@18-15-00fanning.mp4')
results = model.track(source=video_path, show=True)
