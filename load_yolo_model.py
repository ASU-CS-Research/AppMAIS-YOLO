import torch

def load_yolo_model(model_path):
    model = torch.load(model_path)
    return model

if __name__ == '__main__':
    model_path = 'yolov8s.pt'
    model_and_info = load_yolo_model(model_path)
    model = model_and_info['model']
    print(model)
