import numpy as np
import matplotlib.pyplot as plt
import ultralytics
import cv2 as cv
import os
import betabinomial

#writing a function to calculate r squared
def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_tot)

#write a function that plots predicted vs true values
def plot(x, y, x_label, y_label, title, suptitle, save_dest=None):
    # add a color to each point based on the z value

    plt.scatter(x, y, cmap='cool', label='data')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.colorbar()
    plt.title(title)
    plt.suptitle(suptitle)
    #plt.legend(loc='best')
    plt.show()
    if save_dest:
        plt.savefig(save_dest)

if __name__ == "__main__":
    model = ultralytics.YOLO('/home/bee/bee-detection/trained_on_11r_2022.pt')
    path = '/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/split_dataset/val/'

    images_names = os.listdir(f"{path}images/")
    images_names.sort()
    images = [cv.imread(os.path.join(path,"images",image_path)) for image_path in images_names]

    label_files = os.listdir(f"{path}labels/")
    label_files.sort()

    labels = [] # this is in the format [[drones, workers], [drones, workers], ...]

    for label_file in label_files:
        with open(os.path.join(path,"labels",label_file), "r") as f:
            lines = f.readlines()
            label = [0,0]
            for line in lines:
                if line[0] == "0":
                    label[0] += 1
                elif line[0] == "1":
                    label[1] += 1
            labels.append(label)

    predictions = model(images)

    # get the number of drones and workers predicted

    pred = [] # this is in the format [[drones, workers], [drones, workers], ...]

    for prediction in predictions:
        boxes = prediction.boxes
        drones = 0
        workers = 0
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            if class_id == 0:
                drones += 1
            elif class_id == 1:
                workers += 1
        pred.append([drones, workers])

    # get the r squared value for drones
    drones_true = [label[0] for label in labels]
    drones_pred = [label[0] for label in pred]
    r_squared_drones = r_squared(drones_true, drones_pred)
    print("r_squared_drones: ", r_squared_drones)

    # get the r squared value for workers
    workers_true = [label[1] for label in labels]
    workers_pred = [label[1] for label in pred]
    r_squared_workers = r_squared(workers_true, workers_pred)
    print("r_squared_workers: ", r_squared_workers)

    # make a random list of numbers to plot against
    x = np.arange(0, len(drones_true))

    formated_labels = []

    for label_file in label_files:
        #assert os.path.isfile(label_file), f"file {label_file} does not exist"
        with open(f"{path}labels/{label_file}", "r") as f:
            lines = f.readlines()
            image_label = []
            for line in lines:
                bee_label = line.split(" ")
                bee_label = [float(x) for x in bee_label]
                image_label.append(bee_label)

            labels.append(image_label)

    log_likelihoods = betabinomial.beta_binom_on_data(images=images, labels=formated_labels, model=model, image_filenames=images_names)



    # plot the predicted vs true values for drones
    plot(drones_true, drones_pred, "drones_true", "drones_pred", f"drones predicted vs true (r^2 = {r_squared_drones})", "model: trained_on_11s.pt, data: AppMAIS11s_labeled_data val set",  "drones_pred_v_true.png")

    # plot the predicted vs true values for workers
    plot(workers_true, workers_pred , "workers_true", "workers_pred",  f"workers predicted vs true (r^2 = {r_squared_workers})", "model: trained_on_11s.pt, data: AppMAIS11s_labeled_data val set", "workers_pred_v_true.png")

    #numpy function that takes the absolute value of a list


