
import os
import shutil
from sklearn.model_selection import train_test_split


path = "/home/olofintuyita/AppMAIS-YOLO/input_frames"
images = os.listdir(path)

#make 2 folders Will and Tayo
os.mkdir(f"{path}/Will")
os.mkdir(f"{path}/Tayo")

#split the images into 2 equal parts
train, test = train_test_split(images, test_size=0.5, random_state=1)
print("train: ", len(train))
print("test: ", len(test))
#move the train and test images into Tayo and Will folders respectively
for image in train:
    shutil.move(f"{path}/{image}", f"{path}/Tayo/")
for image in test:
    shutil.move(f"{path}/{image}", f"{path}/Will/")







