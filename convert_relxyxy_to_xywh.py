import cv2
# Utility file to convert relative xyxy label format to xywh without changing the labels
image_path = 'data/train/images/0.jpg'
label_path = 'data/train/labels/0.txt'

image = cv2.imread(image_path)
image_height, image_width, _ = image.shape
with open(label_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        label = line.split()
        label[1] = str(float(label[1]) * image_width)
        label[2] = str(float(label[2]) * image_height)
        label[3] = str(float(label[3]) * image_width)
        label[4] = str(float(label[4]) * image_height)
        label = ' '.join(label)
        print(label)
        cv2.rectangle(image, (int(float(label[1])), int(float(label[2]))), (int(float(label[3])), int(float(label[4]))), (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
