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
def plot(x, y, x_label, y_label, title, suptitle, save_dest=None, plot_x_e_y=False, show=False, key = None, alpha=0.6):
    plt.clf()
    font = {#'family': 'normal',
    #         #'weight': 'bold',
            'size': 14}
    plt.rc('font', **font)
    if plot_x_e_y:
        # plot a dashed x=y line to the extent of the greatest x
        plt.plot([0, max(x[0])], [0, max(x[0])], 'k--', label='x=y', linewidth=1)
    # add a color to each point based on the z value
    plt.scatter(x[0], y[0], cmap='cool', label=key[0], alpha=alpha, marker='o')
    plt.scatter(x[1], y[1], cmap='warm', label=key[1], alpha=alpha, marker='^')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    #plt.colorbar()
    plt.title(title)
    plt.suptitle(suptitle)

    if plot_x_e_y:
        plt.legend(loc='best')
    plt.tight_layout()
    if save_dest:
        plt.savefig(save_dest)
    if show:
        plt.show()

if __name__ == "__main__":
    model_path = os.path.abspath("/home/bee/bee-detection/final_model.pt")
    # model_path = os.path.abspath("/home/bee/bee-detection/trained_on_11r_2022.pt")
    model = ultralytics.YOLO(model_path)
    # data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/split_dataset/val/')
    # data_path = os.path.abspath("/home/bee/bee-detection/data_appmais_lab/AppMAIS1s_labeled_data/val/")
    data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/stretch_test_2')
    # data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/final_split_dataset/test')
    # data_path = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS1s_labeled_data/complete_data')
    # data_path = os.path.abspath("/home/bee/bee-detection/data_appmais_lab/AppMAIS1s_labeled_data/train/")
    images_names = os.listdir(os.path.join(data_path, "images"))
    images_names.sort()
    images = [cv.imread(os.path.join(data_path, "images", image_path)) for image_path in images_names]

    label_files = os.listdir(os.path.join(data_path, "labels"))
    label_files.sort()

    labels = []  # this is in the format [[drones, workers], [drones, workers], ...]

    for label_file in label_files:
        if label_file == "classes.txt":
            continue
        with open(os.path.join(data_path, "labels", label_file), "r") as f:
            lines = f.readlines()
            label = [0,0]
            for line in lines:
                if line[0] == "0":
                    label[0] += 1
                elif line[0] == "1":
                    label[1] += 1
            labels.append(label)

    predictions = model.predict(images, conf=0.64)

    # get the number of drones and workers predicted

    pred = []  # this is in the format [[drones, workers], [drones, workers], ...]

    for prediction, image_name, label in zip(predictions, images_names, labels):
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

    all_true = []
    all_pred = []

    for label in labels:
        all_true.append(label[0])
        all_true.append(label[1])
    for label in pred:
        all_pred.append(label[0])
        all_pred.append(label[1])

    r_squared_all = r_squared(all_true, all_pred)

    # make a random list of numbers to plot against
    x = np.arange(0, len(drones_true))

    formated_labels = []

    for label_file in label_files:
        #assert os.data_path.isfile(label_file), f"file {label_file} does not exist"
        if label_file == "classes.txt":
            continue
        with open(os.path.join(data_path, "labels", label_file), "r") as f:
            lines = f.readlines()
            image_label = []
            for line in lines:
                bee_label = line.split(" ")
                bee_label = [float(x) for x in bee_label]
                image_label.append(bee_label)

            labels.append(image_label)

    log_likelihoods = betabinomial.beta_binom_on_data(images=images, labels=formated_labels, model=model,
                                                      image_filenames=images_names)

    # plot the predicted vs true values for drones (seems to look odd on first plt.show(), running twice fixes it)
    # plot(drones_true, drones_pred, "True Drone Count", "Predicted Drone Count",
    #      f"Drone Count Predicted against True (r^2 = {r_squared_drones})",
    #      f"ultralytics_model: {os.path.basename(model_path)}, data: test set",
    #      "drones_pred_v_true.png", show=True)
    #
    # # plot the predicted vs true values for drones
    # plot(drones_true, drones_pred, "True Drone Count", "Predicted Drone Count",
    #      f"Drone Count Predicted against True (r^2 = {r_squared_drones})",
    #      f"ultralytics_model: {os.path.basename(model_path)}, data: Test dataset",
    #      "drones_pred_v_true.png", plot_x_e_y=True, show=True)
    #
    # # plot the predicted vs true values for workers
    # plot(workers_true, workers_pred , "True Worker Count", "Predicted Worker Count",
    #      f"Worker Count Predicted against True (r^2 = {r_squared_workers})",
    #      f"ultralytics_model: {os.path.basename(model_path)}, data: Test dataset",
    #      "workers_pred_v_true.png", plot_x_e_y=True, show=True)

    # plot both on the same graph
    # suptitle can be cropped out for the paper, but it's useful for metric logging
    plot([workers_true, drones_true], [workers_pred, drones_pred], "True Count", "Predicted Count",
         f"Bee Count Predicted against True",
         f"{os.path.basename(model_path)}, {os.path.basename(data_path)}, worker r^2 "
         f"{r_squared_workers: .3f}, drone r^2 {r_squared_drones: .3f}",
         "pred_v_true_both_classes.png", plot_x_e_y=True, show=True, key=["Workers", "Drones"])

    plot([workers_true, drones_true], [workers_pred, drones_pred], "True Count", "Predicted Count",
         f"Bee Count Predicted vs. Actual",
         f"{os.path.basename(model_path)[:-3]}, {os.path.basename(data_path)}, w r^2 "
         f"{r_squared_workers: .3f}, d r^2 {r_squared_drones: .3f}",
         "pred_v_true_both_classes.png", plot_x_e_y=True, show=True, key=["Workers", "Drones"])

