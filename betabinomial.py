import matplotlib.pyplot as plt
from scipy.stats import betabinom, binom, beta
import numpy as np
from typing import List
import ultralytics
import os
import cv2 as cv

def beta_binom_on_data(model, images: List[np.ndarray], labels: np.ndarray) -> List[float]:
    results = model(images)
    log_likelyhoods = []

    for result, label in zip(results, labels):
        bounding_boxes = result.boxes
        # the prior distribution of alpha and beta should not be zero because that would imply that either category does
        # not exist and therefore n should be at least 2 and alpha and beta should be at least 1
        n = 2
        a = 1
        b = 1
        n_true = 2
        a_true = 1
        b_true = 1
        for box in bounding_boxes:
            n += 1
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            a += 1 if class_id == 1 else 0
            b += 1 if class_id == 0 else 0

        p = float(a/n) if n != 0 else 0

        for box in label:
            if len(box) == 0:
                continue

            n_true += 1
            a_true += 1 if box[0] == 1 else 0
            b_true += 1 if box[0] == 0 else 0

        # n_true = 10
        # a_true = 7
        # b_true = n_true - a_true
        p_true = a_true/n_true if n_true != 0 else 0
        n_scaled = n_true * n
        a_scaled = p * n_scaled
        b_scaled = n_scaled - a_scaled

        print("n_predicted: ", n)
        print("a_predicted: ", a)
        print("b_predicted: ", b)
        print("p_predicted: ", p)
        print("n_true: ", n_true)
        print("a_true: ", a_true)
        print("b_true: ", b_true)
        print("p_true: ", p_true)

        #mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

        mean, var, skew, kurt = betabinom.stats(n_scaled, a_scaled, b_scaled, moments='mvsk')

        # print("mean: ", mean)
        # print("var: ", var)
        # print("skew: ", skew)
        # print("kurt: ", kurt)
        # print("p: ", p)

        # plot the distribution
        # x = np.arange(binom.ppf(0.01, n_scaled, p), binom.ppf(0.99, n_scaled, p))
        #
        # plt.plot(x, betabinom.pmf(x, n_scaled, a_scaled, b_scaled), 'bo', ms=8, label='betabinom pmf')
        #plt.show()
        #do the same for a binomial distribution
        # plt.plot(x, binom.pmf(x, n_scaled, p), 'ro', ms=8, label='binom pmf')
        # plt.legend(loc='best')
        #
        # # make a vertical line at the true value of p
        # plt.axvline(x= p_true * n_scaled, color='k', linestyle='--')

        x = n_scaled * p_true
        beta_binom_prob = betabinom.pmf(x, n_scaled, a_scaled, b_scaled)
        # binom_prob = binom.pmf(x, n_scaled, p)
        print("beta_binom_prob: ", beta_binom_prob)
        # print("binom_prob: ", binom_prob)
        # plt.show()

        print("log likelihood beta binom: ", np.log(beta_binom_prob))
        # print("log likelihood binom: ", np.log(binom_prob))

        log_likelyhoods.append(np.log(beta_binom_prob))
        print()

    return log_likelyhoods

if __name__ == "__main__":
    path = "/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/split_dataset/val/"

    labels_filenames = os.listdir(f"{path}labels/")
    labels = []

    model_11s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11s.pt")
    model_1s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_1s.pt")

    images_filenames = os.listdir(f"{path}images/")
    images = [cv.imread(os.path.join(path,"images",image_path)) for image_path in images_filenames]


    for label_filename in labels_filenames:
        with open(f"{path}labels/{label_filename}", "r") as f:
            lines = f.readlines()
            image_label = []
            for line in lines:
                bee_label = line.split(" ")
                bee_label = [float(x) for x in bee_label]
                image_label.append(bee_label)

            labels.append(image_label)


    log_likelihoods_11s = beta_binom_on_data(model_11s, images, labels)
    log_likelihoods_1s = beta_binom_on_data(model_1s, images, labels)
    print(f'On the 11s val set, the mean log likelihood is {np.mean(log_likelihoods_11s)} from the model trained on the '
          f'11s data. The mean log likelihood is {np.mean(log_likelihoods_1s)} from the model trained on the 1s data.')
