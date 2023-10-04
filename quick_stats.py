import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_samples(path):
    # count the number of samples in a directory
    """
    param:
        path: string path to directory containing label files
    return:
        dictionary with keys: images, drones, workers representing the number of images, drones, and workers in the directory
    """

    images = 0
    drones = 0
    workers = 0

    for label in os.listdir(path):
        images += 1
        with open(os.path.join(path, label), "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == "0":
                    drones += 1
                elif line[0] == "1":
                    workers += 1

    return {"images": images,"drones": drones, "workers": workers}



if __name__ == '__main__':
    counts = count_samples("/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/final_split_dataset/test/labels")

    print("images: ", counts["images"])
    print("drones: ", counts["drones"])
    print("workers: ", counts["workers"])

